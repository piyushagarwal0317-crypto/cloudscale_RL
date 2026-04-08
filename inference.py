"""
inference.py — OpenEnv Evaluation Script for CloudScale RL
MANDATORY: Must run in the root of your project.
"""
import asyncio
import os
import sys
import json
import textwrap
import aiohttp
from typing import List, Optional, Dict, Any

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration & Environment Variables
# ---------------------------------------------------------------------------

def get_hf_token() -> str:
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "Missing required environment variable 'HF_TOKEN'. "
            "Please set this variable before running inference.py."
        )
    return token


def get_required_env(var_name: str, purpose: str) -> str:
    value = os.getenv(var_name, "").strip()
    if not value:
        raise SystemExit(
            f"Missing required environment variable '{var_name}'. "
            f"{purpose} Please set this variable before running inference.py."
        )
    return value

HF_TOKEN = get_hf_token()
API_BASE_URL = get_required_env("API_BASE_URL", "This variable must point to your LLM API endpoint.")
MODEL_NAME = get_required_env("MODEL_NAME", "This variable must define the model identifier for inference.")

# Point this to your Hugging Face Space URL or Localhost
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
TASK_NAME = os.getenv("TASK_LEVEL", "medium")  # "easy", "medium", or "hard"
BENCHMARK = "cloudscale_rl"

MAX_STEPS = 200
TEMPERATURE = 0.1 # Low temperature for more deterministic, logical JSON outputs
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.7  # Score required to pass the grader

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an autonomous AI Cloud DevOps Agent managing three microservices: frontend, backend, and worker.
    Your goal is to maintain system health (prevent SLA violations/high latency) while minimizing wasted cloud costs.
    
    Valid actions for each service: "SCALE_UP", "SCALE_DOWN", "NO_OP".
    - Scale UP if queue_depth > 0 or utilization > 0.85
    - Scale DOWN if utilization < 0.4 and queue_depth == 0
    - NO_OP if traffic is stable.
    
    You must respond with a STRICT JSON dictionary matching this exact format:
    {"frontend": "NO_OP", "backend": "NO_OP", "worker": "NO_OP"}
    Do not include markdown tags, reasoning, or any other text. Just the JSON.
    """
).strip()

# ---------------------------------------------------------------------------
# OpenEnv Strict Logging format
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def sanitize_action(action: str) -> str:
    return action.replace("\n", " ").replace("\r", " ").replace('"', "'").strip()


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    action_str = sanitize_action(action)
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

# ---------------------------------------------------------------------------
# LLM Interaction
# ---------------------------------------------------------------------------
def build_user_prompt(step: int, obs: Dict[str, Any]) -> str:
    return textwrap.dedent(
        f"""
        Step: {step}
        Current Cloud State Observation:
        {json.dumps(obs, indent=2)}
        
        Provide your scaling actions as a JSON object.
        """
    ).strip()

def get_model_action(client: OpenAI, step: int, obs: Dict[str, Any]) -> tuple[Dict[str, str], str]:
    user_prompt = build_user_prompt(step, obs)
    raw_text = "{}"
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw_text = (completion.choices[0].message.content or "").strip()
        
        # Strip potential markdown formatting if the model disobeys
        if raw_text.startswith("```json"):
            raw_text = raw_text.strip("`").replace("json\n", "")
            
        action_dict = json.loads(raw_text)
        
        # Ensure all keys exist
        for svc in ["frontend", "backend", "worker"]:
            if svc not in action_dict or action_dict[svc] not in ["SCALE_UP", "SCALE_DOWN", "NO_OP"]:
                action_dict[svc] = "NO_OP"
                
        return action_dict, raw_text
    except Exception as exc:
        print(f"[DEBUG] Model request/parsing failed: {exc}. Raw output: {raw_text}", file=sys.stderr, flush=True)
        # Fallback to safe NO_OPs to keep simulation running
        safe_action = {"frontend": "NO_OP", "backend": "NO_OP", "worker": "NO_OP"}
        return safe_action, json.dumps(safe_action)

# ---------------------------------------------------------------------------
# Async Environment Wrapper
# ---------------------------------------------------------------------------
class CloudEnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = aiohttp.ClientSession()

    async def reset(self, task_level: str) -> Dict[str, Any]:
        async with self.session.post(f"{self.base_url}/reset", json={"task_level": task_level}) as resp:
            data = await resp.json()
            return data["observation"]

    async def step(self, actions: Dict[str, str]) -> Dict[str, Any]:
        async with self.session.post(f"{self.base_url}/step", json={"actions": actions}) as resp:
            return await resp.json()

    async def close(self):
        await self.session.close()

# ---------------------------------------------------------------------------
# Main Evaluation Loop
# ---------------------------------------------------------------------------
async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = CloudEnvClient(ENV_URL)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # 1. Reset Environment
        obs = await env.reset(task_level=TASK_NAME)

        for step in range(1, MAX_STEPS + 1):
            
            # 2. Get LLM Action
            action_dict, raw_action_str = get_model_action(client, step, obs)

            # 3. Step Environment
            result = await env.step(action_dict)
            
            obs = result["observation"]
            done = result["done"]
            info = result["info"]
            
            # Calculate total reward for this step across all agents
            step_rewards = result["rewards"]
            total_step_reward = sum(r["total"] for r in step_rewards.values())
            rewards.append(total_step_reward)
            
            steps_taken = step
            error = None

            # 4. Log Step
            log_step(step=step, action=raw_action_str, reward=total_step_reward, done=done, error=error)

            if done:
                # Retrieve the final grader score we implemented in app.py
                # Clamp to the validator-safe range to ensure score output is valid.
                score = max(0.01, min(0.99, info.get("final_score", 0.0)))
                break

        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", file=sys.stderr, flush=True)
            
        # 5. Log End
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
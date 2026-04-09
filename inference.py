
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

def load_local_env(env_file: str = ".env") -> None:
    """Load KEY=VALUE pairs from a local .env file without overriding existing env vars."""
    if not os.path.isfile(env_file):
        return

    try:
        with open(env_file, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                if key.startswith("export "):
                    key = key[len("export "):].strip()
                if not key or key in os.environ:
                    continue

                value = value.strip().strip('"').strip("'")
                os.environ[key] = value
    except OSError as exc:
        print(f"[DEBUG] Failed to load {env_file}: {exc}", file=sys.stderr, flush=True)


load_local_env()

# Safe defaults — no SystemExit crashes if validator does not set these
HF_TOKEN     = os.getenv("HF_TOKEN",     "").strip()
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1").strip()
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct").strip()

# Warn but do not crash if HF_TOKEN is missing
if not HF_TOKEN:
    print("[WARN] HF_TOKEN is not set. API calls may fail.", file=sys.stderr, flush=True)
    HF_TOKEN = "placeholder"

if not API_BASE_URL:
    print("[WARN] API_BASE_URL is not set. Using default HF inference endpoint.", file=sys.stderr, flush=True)
    API_BASE_URL = "https://api-inference.huggingface.co/v1"

if not MODEL_NAME:
    print("[WARN] MODEL_NAME is not set. Using default model.", file=sys.stderr, flush=True)
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# Point this to your Hugging Face Space URL or Localhost
ENV_URL   = os.getenv("ENV_URL",    "http://localhost:8000")
TASK_NAME = os.getenv("TASK_LEVEL", "medium")  # "easy", "medium", or "hard"
BENCHMARK = "cloudscale_rl"

MAX_STEPS               = 200
TEMPERATURE             = 0.1  # Low temperature for more deterministic, logical JSON outputs
MAX_TOKENS              = 150
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
# OpenEnv Strict Logging Format
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def sanitize_action(action: str) -> str:
    return action.replace("\n", " ").replace("\r", " ").replace('"', "'").strip()


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    action_str = sanitize_action(action)
    error_val  = error if error else "null"
    done_val   = str(done).lower()
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

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
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw_text = (completion.choices[0].message.content or "").strip()

        # Strip potential markdown formatting if the model disobeys
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.startswith("json"):
                raw_text = raw_text[4:].strip()

        action_dict = json.loads(raw_text)

        # Ensure all keys exist and are valid
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
        self.base_url = base_url.rstrip("/")  # Remove trailing slash to avoid double-slash URLs
        self.session  = aiohttp.ClientSession()

    async def reset(self, task_level: str) -> Dict[str, Any]:
        try:
            async with self.session.post(
                f"{self.base_url}/reset",
                json={"task_level": task_level},
            ) as resp:
                data = await resp.json()
                return data["observation"]
        except Exception as exc:
            print(f"[DEBUG] reset() failed: {exc}", file=sys.stderr, flush=True)
            return {}

    async def step(self, actions: Dict[str, str]) -> Dict[str, Any]:
        try:
            async with self.session.post(
                f"{self.base_url}/step",
                json={"actions": actions},
            ) as resp:
                return await resp.json()
        except Exception as exc:
            print(f"[DEBUG] step() failed: {exc}", file=sys.stderr, flush=True)
            # Return safe defaults so the loop can exit cleanly
            return {
                "observation": {},
                "done":        True,
                "info":        {"final_score": 0.0},
                "rewards": {
                    "frontend": {"total": 0.0},
                    "backend":  {"total": 0.0},
                    "worker":   {"total": 0.0},
                },
            }

    async def close(self) -> None:
        try:
            await self.session.close()
        except Exception as exc:
            print(f"[DEBUG] session.close() failed: {exc}", file=sys.stderr, flush=True)

# ---------------------------------------------------------------------------
# Main Evaluation Loop
# ---------------------------------------------------------------------------

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env    = CloudEnvClient(ENV_URL)

    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # 1. Reset Environment
        obs = await env.reset(task_level=TASK_NAME)

        for step in range(1, MAX_STEPS + 1):

            # 2. Get LLM Action
            action_dict, raw_action_str = get_model_action(client, step, obs)

            # 3. Step Environment
            result = await env.step(action_dict)

            obs          = result.get("observation", {})
            done         = result.get("done", False)
            info         = result.get("info", {})
            step_rewards = result.get("rewards", {})

            # Safely calculate total reward — handles missing keys gracefully
            total_step_reward = sum(
                r.get("total", 0.0)
                for r in step_rewards.values()
                if isinstance(r, dict)
            )
            rewards.append(total_step_reward)

            steps_taken = step

            # 4. Log Step
            log_step(step=step, action=raw_action_str, reward=total_step_reward, done=done, error=None)

            if done:
                # Clamp to validator-safe range
                score = max(0.01, min(0.99, info.get("final_score", 0.0)))
                break

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Unhandled error in evaluation loop: {exc}", file=sys.stderr, flush=True)

    finally:
        await env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"[DEBUG] Fatal error in main: {exc}", file=sys.stderr, flush=True)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        sys.exit(1)

#!/usr/bin/env python3
"""
train_autoscaler.py — GRPO Training Script for the Cloud Autoscaling Agent
===============================================================================
Trains a small LLM to act as an autonomous Site Reliability Engineer (SRE) that 
monitors microservices and scales pods in response to flash traffic spikes.

Follows the OpenEnv rollout pattern:
  • MockCloudEnv fallback    (offline demo / Colab — no server needed)
  • rollout_func + rollout_once  (multi-turn episode collection)
  • Multi-level reward functions (environment + format + reasoning)
  • GRPOTrainer from TRL         

Training Flow
-------------
    1. Build prompt dataset from cloud scenarios (simulated RPS, queues, latency).
    2. For each prompt: generate scaling action → step environment → collect reward.
    3. Multi-level rewards guide the agent to learn:
         - Valid JSON action format (SCALE_UP, SCALE_DOWN, NO_OP)
         - Correct scaling decisions  (env_reward)
         - SRE Chain-of-Thought       (reasoning_reward)
    4. GRPO optimises the policy.
"""
import json
import random
import re
from typing import Dict, List, Any, Optional

from datasets import Dataset
from transformers import AutoTokenizer

# ==========================================================================
# 1. Environment Connection (Mock Cloud Env for standalone training)
# ==========================================================================
class MockCloudEnv:
    """
    Lightweight mock that mimics the CloudAutoScalerEnv interface.
    Generates synthetic cloud monitoring observations so the training script 
    runs completely standalone in Colab without needing the FastAPI server.
    """
    SERVICES = ["frontend", "backend", "worker"]

    class Obs:
        """Mock observation — mirrors CloudObservation fields."""
        def __init__(self, round_num: int, max_rounds: int):
            self.round_number = round_num
            self.max_rounds   = max_rounds
            
            # Determine if we are simulating a flash spike
            is_spike = random.random() < 0.2
            self.spike_multiplier = random.uniform(2.5, 5.0) if is_spike else 1.0
            
            self.data = {
                "time_step": round_num,
                "max_steps": max_rounds,
                "active_alerts": [{"pattern": "flash_spike", "intensity_multiplier": self.spike_multiplier}] if is_spike else [],
                "global_stats": {
                    "uptime_seconds": round_num * 5,
                    "active_spikes": 1 if is_spike else 0,
                    "sla_violations": random.randint(0, 2) if is_spike else 0
                }
            }

            # Generate metrics for a specific target service for this prompt
            self.target_service = random.choice(MockCloudEnv.SERVICES)
            utilization = random.uniform(0.1, 0.95)
            
            # Force high utilization and latency if there's a spike
            if is_spike:
                utilization = random.uniform(0.85, 1.0)
                queue = random.uniform(10.0, 50.0)
                latency = random.uniform(150.0, 400.0)
            else:
                queue = 0.0 if utilization < 0.8 else random.uniform(0.0, 5.0)
                latency = random.uniform(10.0, 90.0) if queue == 0 else random.uniform(80.0, 200.0)

            self.data[self.target_service] = {
                "active_pods": random.randint(2, 20),
                "max_replicas": 50,
                "pending_pods": 0 if random.random() < 0.8 else random.randint(1, 3),
                "utilization": utilization,
                "queue_depth": queue,
                "latency_p95": latency,
                "error_rate": 0.05 if latency > 300 else 0.0,
                "rps": random.uniform(100, 1000) * self.spike_multiplier,
            }

        def build_prompt(self) -> str:
            d = self.data
            svc = d[self.target_service]
            alerts_text = "YES (Flash Spike)" if d["active_alerts"] else "None (Stable)"
            
            return (
                f"Round {d['time_step']}/{d['max_steps']}. You are managing the '{self.target_service}' service.\n"
                f"\nSERVICE METRICS:"
                f"\n- Active Pods : {svc['active_pods']} / {svc['max_replicas']} (Pending: {svc['pending_pods']})"
                f"\n- CPU Util.   : {svc['utilization']*100:.1f}%"
                f"\n- Queue Depth : {svc['queue_depth']:.1f}"
                f"\n- P95 Latency : {svc['latency_p95']:.1f}ms"
                f"\n- Error Rate  : {svc['error_rate']:.2f}"
                f"\n\nTRAFFIC:"
                f"\n- Current RPS : {svc['rps']:.1f}"
                f"\n- Spikes?     : {alerts_text}"
                f"\n\nDecide your scaling action as JSON."
            )

    class Result:
        def __init__(self, obs, reward, done):
            self.observation = obs.data
            self.reward      = reward
            self.done        = done

    def __init__(self):
        self._round = 0
        self._max_rounds = 6

    def reset(self, **kwargs) -> "MockCloudEnv.Result":
        self._round = 0
        obs = self.Obs(self._round, self._max_rounds)
        return self.Result(obs, 0.0, False)

    def step(self, action_input, obs: "MockCloudEnv.Obs") -> "MockCloudEnv.Result":
        self._round += 1
        next_obs = self.Obs(self._round, self._max_rounds)
        reward = self._compute_reward(action_input, obs)
        done = self._round >= self._max_rounds
        return self.Result(next_obs, reward, done)

    def _compute_reward(self, action_input, obs: "MockCloudEnv.Obs") -> float:
        """Scores the scaling action against the synthetic cloud state."""
        try:
            if isinstance(action_input, str):
                match = re.search(r'\{[^}]+\}', action_input, re.DOTALL)
                parsed = json.loads(match.group()) if match else json.loads(action_input)
            else:
                parsed = action_input

            action = parsed.get("action_type", "NO_OP")
            svc = obs.data[obs.target_service]
            has_spike = len(obs.data["active_alerts"]) > 0
            
            # 1. Spike Handling Logic
            if has_spike:
                if action == "SCALE_UP": return 0.8     # Perfect response
                if action == "NO_OP": return -0.4       # Ignored the spike
                if action == "SCALE_DOWN": return -0.8  # Catastrophic failure
                
            # 2. Stable Load Logic
            util = svc["utilization"]
            if action == "SCALE_DOWN":
                if util < 0.4: return 0.5   # Good cost savings
                if util > 0.7: return -0.6  # Dangerous scale down
            
            if action == "SCALE_UP":
                if util > 0.8 or svc["queue_depth"] > 5: return 0.5  # Good scale up
                if util < 0.5: return -0.4                           # Wasteful scale up
                
            if action == "NO_OP":
                if 0.4 <= util <= 0.8 and svc["queue_depth"] == 0: return 0.4  # Good stability
                if util > 0.85: return -0.3                                    # Should have scaled up

        except Exception:
            return -0.5 # Malformed JSON penalty

        return 0.0

env = MockCloudEnv()

# ==========================================================================
# 2. System Prompt & Dataset
# ==========================================================================
SYSTEM_PROMPT = """You are an autonomous Site Reliability Engineering (SRE) agent managing a microservice.
Your goal is to maintain strict SLAs (P95 Latency < 100ms) while minimizing costs.
Anticipate traffic spikes and scale proactively.

RESPOND WITH EXACTLY ONE JSON OBJECT. Valid action_types:
  SCALE_UP   — Provision an additional pod (use when util is high or spikes occur)
  SCALE_DOWN — Terminate a pod (use when util is low and traffic is stable)
  NO_OP      — Maintain current capacity

EXAMPLES:
  {"action_type": "SCALE_UP", "reason": "Flash spike detected and CPU utilization is at 88%. Scaling up to prevent queue buildup."}
  {"action_type": "SCALE_DOWN", "reason": "System is stable with no spikes and CPU util is 35%. Scaling down to save costs."}
  {"action_type": "NO_OP", "reason": "CPU util is optimal at 65% with no latency issues. Maintaining current capacity."}

Your goal: Prevent SLA violations during spikes, and save money during quiet periods."""

def make_prompts(n: int = 64) -> List[Dict[str, str]]:
    rows = []
    for _ in range(n):
        obs = MockCloudEnv.Obs(random.randint(1, 50), 100)
        rows.append({"prompt": obs.build_prompt()})
    return rows

dataset = Dataset.from_list(make_prompts(256))

# ==========================================================================
# 3. Rollout Function & Rewards
# ==========================================================================
def rollout_func(prompts: list, trainer) -> dict:
    episode_prompt_ids = []
    episode_completion_ids = []
    episode_logprobs = []
    episode_env_masks = []
    env_rewards = []
    format_rewards = []
    reasoning_rewards = []
    tokenizer = trainer.processing_class

    for prompt_text in prompts:
        episode = rollout_once(trainer, env, tokenizer, prompt_text, SYSTEM_PROMPT, max_turns=6, max_new_tokens=192)
        episode_prompt_ids.append(episode["prompt_ids"])
        episode_completion_ids.append(episode["completion_ids"])
        episode_logprobs.append(episode["logprobs"])
        episode_env_masks.append(episode["env_mask"])
        env_rewards.append(episode["env_reward"])
        format_rewards.append(episode["format_reward"])
        reasoning_rewards.append(episode["reasoning_reward"])

    return {
        "prompt_ids": episode_prompt_ids,
        "completion_ids": episode_completion_ids,
        "logprobs": episode_logprobs,
        "env_mask": episode_env_masks,
        "env_reward": env_rewards,
        "format_reward": format_rewards,
        "reasoning_reward": reasoning_rewards,
    }

def rollout_once(trainer, env, tokenizer, prompt_text, system_prompt, max_turns, max_new_tokens) -> dict:
    from trl.experimental.openenv import generate_rollout_completions
    
    result = env.reset()
    current_obs = result.observation # Track obs specifically for reward computation
    
    prompt_ids, completion_ids, logprobs, env_mask, model_outputs, raw_rewards = [], [], [], [], [], []
    accumulated_messages = [{"role": "system", "content": system_prompt}]

    initial_text = tokenizer.apply_chat_template(
        accumulated_messages + [{"role": "user", "content": prompt_text}],
        add_generation_prompt=True, tokenize=False
    )
    prompt_ids.extend(tokenizer.encode(initial_text, add_special_tokens=False))

    for _ in range(max_turns):
        if result.done: break

        messages = accumulated_messages + [{"role": "user", "content": prompt_text}]
        full_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        rollout_outputs = generate_rollout_completions(trainer, [full_prompt], generation_overrides={"max_tokens": max_new_tokens})[0]
        
        nl_tokens = tokenizer.encode("\n", add_special_tokens=False)
        completion_ids.extend(nl_tokens + rollout_outputs["completion_ids"] + nl_tokens)
        logprobs.extend([0.0]*len(nl_tokens) + rollout_outputs["logprobs"] + [0.0]*len(nl_tokens))
        env_mask.extend([1]*(len(nl_tokens) * 2 + len(rollout_outputs["completion_ids"])))

        completion_text = rollout_outputs.get("text") or tokenizer.decode(rollout_outputs["completion_ids"], skip_special_tokens=True)
        model_outputs.append(completion_text.strip())

        # Crucial: Step env passing the *previous* obs so the mock knows what state the LLM reacted to
        result = env.step(completion_text, current_obs) 
        current_obs = result.observation
        
        raw_rewards.append(float(result.reward or 0.0))

        env_feedback = f"\nReward: {result.reward:+.2f}."
        env_tokens = tokenizer.encode(env_feedback, add_special_tokens=False)
        completion_ids.extend(env_tokens)
        logprobs.extend([0.0] * len(env_tokens))
        env_mask.extend([0] * len(env_tokens)) # Mask env feedback from loss

        accumulated_messages.append({"role": "user", "content": prompt_text})
        accumulated_messages.append({"role": "assistant", "content": completion_text + env_feedback})

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "env_mask": env_mask,
        "env_reward": sum(raw_rewards) / max(len(raw_rewards), 1),
        "format_reward": compute_format_reward(model_outputs),
        "reasoning_reward": compute_reasoning_reward(model_outputs),
        "model_outputs": model_outputs,
    }

# --------------------------------------------------------------------------
# 4. Reward Calculation Methods
# --------------------------------------------------------------------------
def compute_format_reward(model_outputs: List[str]) -> float:
    if not model_outputs: return 0.0
    valid_types = {"SCALE_UP", "SCALE_DOWN", "NO_OP"}
    correct = 0
    for text in model_outputs:
        try:
            match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if match and json.loads(match.group()).get("action_type") in valid_types:
                correct += 1
        except Exception: pass
    return correct / len(model_outputs)

def compute_reasoning_reward(model_outputs: List[str]) -> float:
    if not model_outputs: return 0.0
    score = 0.0
    for text in model_outputs:
        try:
            match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if match:
                reason = json.loads(match.group()).get("reason", "")
                if len(reason) > 15: score += 0.5
                if "utilization" in reason.lower() or "spike" in reason.lower() or "latency" in reason.lower():
                    score += 0.5 # Bonus for referencing domain logic
        except Exception: pass
    return min(1.0, score / len(model_outputs))

def reward_env(completions, **kwargs): return [float(r) for r in kwargs.get("env_reward", [])] or [0.0]*len(completions)
def reward_format(completions, **kwargs): return [float(r) for r in kwargs.get("format_reward", [])] or [0.0]*len(completions)
def reward_reasoning(completions, **kwargs): return [float(r) for r in kwargs.get("reasoning_reward", [])] or [0.0]*len(completions)

# ==========================================================================
# 5. Configure and Launch GRPO Training
# ==========================================================================
if __name__ == "__main__":
    from trl import GRPOConfig, GRPOTrainer
    
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    output_dir = "cloudscale-autoscaler-agent"

    print("=" * 60)
    print("CloudScale Ops — GRPO AutoScaler Agent Training")
    print(f"Model      : {model_id}")
    print(f"Dataset    : {len(dataset)} synthetic cloud spikes")
    print("=" * 60)

    grpo_config = GRPOConfig(
        num_train_epochs=1,
        learning_rate=1e-6,
        gradient_accumulation_steps=16,
        per_device_train_batch_size=1,
        max_grad_norm=1.0,
        num_generations=2,
        max_completion_length=1024,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.15,
        output_dir=output_dir,
        report_to="none",
        logging_steps=1,
        save_steps=50,
        gradient_checkpointing=True,
        push_to_hub=False,
    )

    trainer = GRPOTrainer(
        model=model_id,
        processing_class=tokenizer,
        reward_funcs=[reward_env, reward_format, reward_reasoning],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
    )

    trainer.train()
    print(f"\nTraining complete! Agent saved to ./{output_dir}")

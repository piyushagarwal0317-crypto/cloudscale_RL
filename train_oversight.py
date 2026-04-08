#!/usr/bin/env python3
"""
train_oversight.py — GRPO Training Script for the Global Orchestrator
===============================================================================
Trains a small LLM to act as the Oversight Agent. It receives the requested 
scaling actions from the individual microservice agents (frontend, backend, worker) 
and must resolve conflicts (e.g., preventing a downstream scale-down when upstream 
scales up) and enforce global pod budgets.
"""
import json
import random
import re
from typing import Dict, List

from datasets import Dataset
from transformers import AutoTokenizer

# ==========================================================================
# 1. Environment Connection (Mock Oversight Env)
# ==========================================================================
class MockOversightEnv:
    """
    Generates synthetic multi-agent proposals to train the Oversight orchestrator
    on conflict resolution and budget constraints.
    """
    SERVICES = ["frontend", "backend", "worker"]
    MAX_GLOBAL_PODS = 50

    class Obs:
        def __init__(self, round_num: int, max_rounds: int):
            self.round_number = round_num
            self.max_rounds = max_rounds
            
            # Generate random current pod counts
            self.current_pods = {
                "frontend": random.randint(5, 20),
                "backend": random.randint(5, 20),
                "worker": random.randint(5, 10)
            }
            self.total_pods = sum(self.current_pods.values())
            
            # Generate random proposals from the swarm
            actions = ["SCALE_UP", "SCALE_DOWN", "NO_OP"]
            self.proposals = {
                "frontend": random.choice(actions),
                "backend": random.choice(actions),
                "worker": random.choice(actions)
            }
            
            # Manually inject conflicts 30% of the time to force the model to learn
            if random.random() < 0.3:
                self.proposals["frontend"] = "SCALE_UP"
                self.proposals["backend"] = "SCALE_DOWN" # Bottleneck conflict!
                
            # Inject budget crises 20% of the time
            if random.random() < 0.2:
                self.current_pods["frontend"] = 25
                self.current_pods["backend"] = 20
                self.current_pods["worker"] = 4
                self.total_pods = 49
                self.proposals["frontend"] = "SCALE_UP"
                self.proposals["backend"] = "SCALE_UP" # Budget overrun conflict!

        def build_prompt(self) -> str:
            return (
                f"Round {self.round_number}/{self.max_rounds}. You are the Oversight Agent.\n\n"
                f"GLOBAL CLUSTER STATE:\n"
                f"- Global Pod Budget : {self.total_pods} / {MockOversightEnv.MAX_GLOBAL_PODS}\n"
                f"- Frontend Pods   : {self.current_pods['frontend']}\n"
                f"- Backend Pods    : {self.current_pods['backend']}\n"
                f"- Worker Pods     : {self.current_pods['worker']}\n\n"
                f"AGENT PROPOSALS:\n"
                f"- frontend wants to : {self.proposals['frontend']}\n"
                f"- backend wants to  : {self.proposals['backend']}\n"
                f"- worker wants to   : {self.proposals['worker']}\n\n"
                f"Approve or override these actions. Respond as a JSON dictionary."
            )

    class Result:
        def __init__(self, obs, reward, done):
            self.observation = obs
            self.reward = reward
            self.done = done

    def __init__(self):
        self._round = 0
        self._max_rounds = 6

    def reset(self, **kwargs) -> "MockOversightEnv.Result":
        self._round = 0
        obs = self.Obs(self._round, self._max_rounds)
        return self.Result(obs, 0.0, False)

    def step(self, action_input, obs: "MockOversightEnv.Obs") -> "MockOversightEnv.Result":
        self._round += 1
        next_obs = self.Obs(self._round, self._max_rounds)
        reward = self._compute_reward(action_input, obs)
        done = self._round >= self._max_rounds
        return self.Result(next_obs, reward, done)

    def _compute_reward(self, action_input, obs: "MockOversightEnv.Obs") -> float:
        """Scores the orchestrator on catching budget limits and bottlenecks."""
        try:
            if isinstance(action_input, str):
                match = re.search(r'\{[^}]+\}', action_input, re.DOTALL)
                parsed = json.loads(match.group()) if match else json.loads(action_input)
            else:
                parsed = action_input

            final_actions = parsed.get("actions", {})
            if not isinstance(final_actions, dict) or not all(s in final_actions for s in self.SERVICES):
                return -0.5 # Bad format

            reward = 0.0
            f_act = final_actions.get("frontend")
            b_act = final_actions.get("backend")
            w_act = final_actions.get("worker")

            # Rule 1: Bottleneck Prevention
            if obs.proposals["frontend"] == "SCALE_UP" and obs.proposals["backend"] == "SCALE_DOWN":
                if b_act == "SCALE_DOWN":
                    reward -= 1.0 # Failed to catch bottleneck
                elif b_act == "NO_OP":
                    reward += 1.0 # Successfully overridden

            # Rule 2: Budget Enforcement
            projected_additions = sum(1 for act in [f_act, b_act, w_act] if act == "SCALE_UP")
            projected_removals = sum(1 for act in [f_act, b_act, w_act] if act == "SCALE_DOWN")
            new_total = obs.total_pods + projected_additions - projected_removals
            
            if new_total > self.MAX_GLOBAL_PODS:
                reward -= 1.0 # Allowed budget overrun!
            elif obs.total_pods >= self.MAX_GLOBAL_PODS - 1 and projected_additions > 0 and new_total <= self.MAX_GLOBAL_PODS:
                reward += 1.0 # Successfully blocked an overrun

            # Rule 3: Don't meddle if not necessary
            if reward == 0.0: 
                # If no conflicts existed, did they just pass through the proposals?
                if f_act == obs.proposals["frontend"] and b_act == obs.proposals["backend"] and w_act == obs.proposals["worker"]:
                    reward += 0.5

            return reward

        except Exception:
            return -0.5 # Malformed JSON penalty

env = MockOversightEnv()

# ==========================================================================
# 2. System Prompt & Dataset
# ==========================================================================
SYSTEM_PROMPT = """You are the Global Oversight Agent for a microservice cluster.
You receive scaling proposals from the frontend, backend, and worker agents.
You MUST apply these two SRE rules:
1. BOTTLENECK RULE: If frontend scales UP, downstream services (backend, worker) CANNOT scale DOWN. Change their action to NO_OP.
2. BUDGET RULE: The global pod count must never exceed 50. If proposed SCALE_UPs exceed this, change them to NO_OP.

RESPOND WITH EXACTLY ONE JSON OBJECT containing the final approved actions.

EXAMPLE:
{
  "actions": {
    "frontend": "SCALE_UP",
    "backend": "NO_OP",
    "worker": "NO_OP"
  },
  "reason": "Frontend scale up approved. Backend scale down overridden to prevent bottlenecks."
}"""

def make_prompts(n: int = 64) -> List[Dict[str, str]]:
    rows = []
    for _ in range(n):
        obs = MockOversightEnv.Obs(random.randint(1, 50), 100)
        rows.append({"prompt": obs.build_prompt()})
    return rows

dataset = Dataset.from_list(make_prompts(256))

# ==========================================================================
# 3. Rollout Function & Rewards
# ==========================================================================
def rollout_func(prompts: list, trainer) -> dict:
    episode_prompt_ids, episode_completion_ids, episode_logprobs, episode_env_masks = [], [], [], []
    env_rewards, format_rewards = [], []
    tokenizer = trainer.processing_class

    for prompt_text in prompts:
        episode = rollout_once(trainer, env, tokenizer, prompt_text, SYSTEM_PROMPT, max_turns=6, max_new_tokens=192)
        episode_prompt_ids.append(episode["prompt_ids"])
        episode_completion_ids.append(episode["completion_ids"])
        episode_logprobs.append(episode["logprobs"])
        episode_env_masks.append(episode["env_mask"])
        env_rewards.append(episode["env_reward"])
        format_rewards.append(episode["format_reward"])

    return {
        "prompt_ids": episode_prompt_ids,
        "completion_ids": episode_completion_ids,
        "logprobs": episode_logprobs,
        "env_mask": episode_env_masks,
        "env_reward": env_rewards,
        "format_reward": format_rewards,
    }

def rollout_once(trainer, env, tokenizer, prompt_text, system_prompt, max_turns, max_new_tokens) -> dict:
    from trl.experimental.openenv import generate_rollout_completions
    
    result = env.reset()
    current_obs = result.observation
    
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

        result = env.step(completion_text, current_obs) 
        current_obs = result.observation
        
        raw_rewards.append(float(result.reward or 0.0))

        env_feedback = f"\nReward: {result.reward:+.2f}."
        env_tokens = tokenizer.encode(env_feedback, add_special_tokens=False)
        completion_ids.extend(env_tokens)
        logprobs.extend([0.0] * len(env_tokens))
        env_mask.extend([0] * len(env_tokens))

        accumulated_messages.append({"role": "user", "content": prompt_text})
        accumulated_messages.append({"role": "assistant", "content": completion_text + env_feedback})

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "env_mask": env_mask,
        "env_reward": sum(raw_rewards) / max(len(raw_rewards), 1),
        "format_reward": compute_format_reward(model_outputs),
        "model_outputs": model_outputs,
    }

def compute_format_reward(model_outputs: List[str]) -> float:
    if not model_outputs: return 0.0
    correct = 0
    for text in model_outputs:
        try:
            match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                if "actions" in data and isinstance(data["actions"], dict):
                    correct += 1
        except Exception: pass
    return correct / len(model_outputs)

def reward_env(completions, **kwargs): return [float(r) for r in kwargs.get("env_reward", [])] or [0.0]*len(completions)
def reward_format(completions, **kwargs): return [float(r) for r in kwargs.get("format_reward", [])] or [0.0]*len(completions)

# ==========================================================================
# 4. Configure and Launch GRPO Training
# ==========================================================================
if __name__ == "__main__":
    from trl import GRPOConfig, GRPOTrainer
    
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    output_dir = "cloudscale-oversight-agent"

    print("=" * 60)
    print("CloudScale Ops — GRPO Oversight Agent Training")
    print(f"Model      : {model_id}")
    print(f"Dataset    : {len(dataset)} synthetic multi-agent conflicts")
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
        reward_funcs=[reward_env, reward_format],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
    )

    trainer.train()
    print(f"\nTraining complete! Oversight Agent saved to ./{output_dir}")
from __future__ import annotations

import os
import time
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Import the updated environment
from server.cloudscale_RL_environment import CloudAutoScalerEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CloudScale Admin — OpenEnv API",
    version="1.0.0",
    description="Scalable RL Environment for microservice autoscaling and traffic spike management.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_STEPS = int(os.environ.get("MAX_STEPS", "200"))
_env: Optional[CloudAutoScalerEnv] = None
_last_obs = None

def get_env(task_level: str = "medium") -> CloudAutoScalerEnv:
    global _env
    # Re-initialize if the environment doesn't exist OR if the task level changed
    if _env is None or _env.task_level != task_level:
        _env = CloudAutoScalerEnv(
            task_level=task_level,
            max_steps=MAX_STEPS
        )
    return _env

# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------
class StepRequest(BaseModel):
    actions: Dict[str, str] = Field(
        default={"frontend": "NO_OP", "backend": "NO_OP", "worker": "NO_OP"},
        description="Scaling actions for each microservice agent."
    )

class ResetRequest(BaseModel):
    task_level: str = Field(
        default="medium",
        description="Difficulty of the task: 'easy', 'medium', or 'hard'."
    )

# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------
def _obs_to_dict(obs_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        agent_id: {
            "active_pods": obs.get("active_pods"),
            "pending_pods": obs.get("pending_pods"),
            "rps": obs.get("rps"),
            "latency_p95": obs.get("latency_p95"),
            "error_rate": obs.get("error_rate"),
            "queue_depth": obs.get("queue_depth"),
            "utilization": obs.get("utilization"),
            "spike_detected": obs.get("spike_detected")
        }
        for agent_id, obs in obs_dict.items()
    }

def _reward_to_dict(rewards_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        agent_id: {
            "total": reward.get("total", 0.0),
            "components": {
                "r_latency": reward.get("r_latency", 0.0),
                "r_cost": reward.get("r_cost", 0.0),
                "r_action": reward.get("r_action", 0.0),
                "r_spike": reward.get("r_spike", 0.0),
                "r_sla": reward.get("r_sla", 0.0)
            }
        }
        for agent_id, reward in rewards_dict.items()
    }

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    env = get_env()
    return {
        "status": "ok",
        "system_healthy": env.stats.system_healthy,
        "uptime": time.time() - env.stats.start_time,
        "timestamp": time.time(),
    }

@app.post("/reset")
def reset(req: ResetRequest = None):
    global _last_obs
    # Default to medium if no request body is provided
    task_level = req.task_level if req else "medium"
    
    env = get_env(task_level=task_level)
    obs = env.reset()
    _last_obs = obs
    return {
        "status": "reset", 
        "task_level": task_level,
        "observation": _obs_to_dict(obs)
    }

@app.post("/step")
def step(req: StepRequest):
    global _last_obs
    env = get_env()
    
    if _last_obs is None:
        env.reset()
        
    # FIX: Correctly unpack the CloudStepResult object
    result = env.step(req.actions)
    _last_obs = result.observations
    
    info = result.info
    
    # 🏆 KILLER FEATURE: Inject the Grader Score when the episode finishes
    if result.done:
        final_score = env.grade_task()
        final_score = max(0.01, min(0.99, final_score))
        info["final_score"] = final_score
        info["task_level"] = env.task_level
        logger.info(f"Episode finished. Task: {env.task_level} | Final Score: {info['final_score']}")
    
    return {
        "done": result.done,
        "info": info,
        "observation": _obs_to_dict(result.observations),
        "rewards": _reward_to_dict(result.rewards),
    }

@app.get("/state")
def get_state():
    env = get_env()
    obs = env.get_global_state()
    return _obs_to_dict(obs)
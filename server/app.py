from __future__ import annotations

import os
import time
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure the parent directory is on sys.path when running from server/
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Import your new Cloud environment models
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

# ---------------------------------------------------------------------------
# Global environment (singleton, recreated on /reset)
# ---------------------------------------------------------------------------
MAX_STEPS = int(os.environ.get("MAX_STEPS", "200"))
BASE_LOAD = int(os.environ.get("BASE_LOAD", "500"))

_env: Optional[CloudAutoScalerEnv] = None
_last_obs = None

def get_env() -> CloudAutoScalerEnv:
    global _env
    if _env is None:
        # Initialize the environment with cloud-specific configs
        _env = CloudAutoScalerEnv(
            max_steps=MAX_STEPS,
            base_load=BASE_LOAD,
            spike_probability=0.05
        )
    return _env

# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------
class StepRequest(BaseModel):
    # Maps agent_id (service name) to an action (SCALE_UP, SCALE_DOWN, NO_OP)
    actions: Dict[str, str] = Field(
        default={"frontend": "NO_OP", "backend": "NO_OP", "worker": "NO_OP"},
        description="Scaling actions for each microservice agent."
    )

# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------
def _obs_to_dict(obs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Converts the multi-agent observation space into JSON-safe dicts."""
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
    """Serializes the strict, discrete bucketed rewards."""
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

def _system_stats_to_dict(stats) -> Dict[str, Any]:
    """Global system metrics."""
    return {
        "global_time_step": stats.global_time,
        "total_requests_processed": stats.total_requests,
        "total_sla_violations": stats.sla_violations,
        "active_spikes": stats.active_spikes,
        "uptime_seconds": time.time() - stats.start_time,
        "system_healthy": stats.system_healthy
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
def reset():
    global _last_obs
    env = get_env()
    obs = env.reset()
    _last_obs = obs
    return {
        "status": "reset", 
        "observation": _obs_to_dict(obs)
    }

@app.post("/step")
def step(req: StepRequest):
    global _last_obs
    env = get_env()
    
    if _last_obs is None:
        # Auto-reset on first step if not explicitly called
        env.reset()
        
    # Pass the multi-agent action dictionary to the environment
    obs, rewards, done, info = env.step(req.actions)
    _last_obs = obs
    
    return {
        "done": done,
        "info": info,
        "observation": _obs_to_dict(obs),
        "rewards": _reward_to_dict(rewards),
    }

@app.get("/state")
def get_state():
    """Returns a snapshot of the current state without advancing the simulation."""
    env = get_env()
    obs = env.get_global_state()
    return _obs_to_dict(obs)

@app.get("/metrics")
def get_metrics():
    """System-wide monitoring stats (replaces the old /stats endpoint)."""
    env = get_env()
    return _system_stats_to_dict(env.stats)

@app.get("/services")
def get_services():
    """Returns configuration and limits for all active microservices."""
    env = get_env()
    return {
        name: {
            "min_replicas": svc.min_replicas,
            "max_replicas": svc.max_replicas,
            "current_replicas": svc.active_pods
        }
        for name, svc in env.services.items()
    }

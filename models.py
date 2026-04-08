"""
models.py — Data models for the Cloud Autoscaling environment.

Strictly adheres to the OpenEnv Pydantic specification, mapping the 
multi-agent cloud simulation to the client and FastAPI backend.
"""
from pydantic import BaseModel, Field
from typing import Dict
from enum import Enum

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class ActionType(str, Enum):
    SCALE_UP   = "SCALE_UP"
    SCALE_DOWN = "SCALE_DOWN"
    NO_OP      = "NO_OP"

# ---------------------------------------------------------------------------
# Core OpenEnv Models (Required by client.py & app.py)
# ---------------------------------------------------------------------------

class CloudscaleRlAction(BaseModel):
    """
    The multi-agent action space. 
    Maps the service name to the specific scaling action.
    """
    actions: Dict[str, ActionType] = Field(
        default={
            "frontend": ActionType.NO_OP, 
            "backend": ActionType.NO_OP, 
            "worker": ActionType.NO_OP
        },
        description="Scaling actions for each microservice agent."
    )

class AgentObservation(BaseModel):
    """Real-time performance metrics for a single microservice."""
    active_pods: float
    pending_pods: float
    rps: float
    latency_p95: float
    error_rate: float
    queue_depth: float
    utilization: float
    spike_detected: float

class CloudscaleRlObservation(BaseModel):
    """
    The full global observation space containing all microservices.
    This exactly matches the JSON output from the FastAPI /step endpoint.
    """
    frontend: AgentObservation
    backend: AgentObservation
    worker: AgentObservation

# ---------------------------------------------------------------------------
# Extended/Internal Models (Optional extensions for future use)
# ---------------------------------------------------------------------------

class RewardComponent(BaseModel):
    """Strict discrete bucketing reward breakdown."""
    r_latency: float = 0.0
    r_cost: float = 0.0
    r_action: float = 0.0
    r_spike: float = 0.0
    r_sla: float = 0.0

class AgentReward(BaseModel):
    """Total reward and component breakdown for a single service."""
    total: float = 0.0
    components: RewardComponent

class CloudscaleRlRewards(BaseModel):
    """The multi-agent reward structure returned by the environment."""
    frontend: AgentReward
    backend: AgentReward
    worker: AgentReward
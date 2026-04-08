"""
models.py — Data models for the Cloud Autoscaling environment.

All data flowing through the OpenEnv simulation, workload generator, 
and RL/LLM agents passes through these strictly typed dataclasses.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
import time

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class ActionType(str, Enum):
    SCALE_UP   = "SCALE_UP"
    SCALE_DOWN = "SCALE_DOWN"
    NO_OP      = "NO_OP"

class TrafficPattern(str, Enum):
    NORMAL      = "normal"
    FLASH_SPIKE = "flash_spike"    # Sudden >2x RPS jump
    DIURNAL     = "diurnal"        # Predictable day/night wave
    DECAY       = "decay"          # Post-spike traffic drop-off

class HealthStatus(str, Enum):
    HEALTHY  = "healthy"           # P95 latency < 100ms
    DEGRADED = "degraded"          # High latency, queue buildup
    CRITICAL = "critical"          # SLA violated, error rate > 0

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ServiceMetrics:
    """Real-time performance metrics for a single microservice."""
    service_id: str
    active_pods: int              = 0
    pending_pods: int             = 0
    max_replicas: int             = 0
    min_replicas: int             = 0
    throughput_per_pod: float     = 0.0
    rps: float                    = 0.0
    utilization: float            = 0.0
    queue_depth: float            = 0.0
    latency_p95: float            = 0.0
    error_rate: float             = 0.0
    health: HealthStatus          = HealthStatus.HEALTHY
    last_updated: float           = field(default_factory=time.time)

@dataclass
class TrafficAlert:
    """A flag raised when a flash spike is detected by the workload generator."""
    alert_id: str
    timestamp: float
    service_id: str
    pattern: TrafficPattern
    intensity_multiplier: float   # e.g., 4.0 for a 400% traffic jump
    description: str
    acknowledged: bool            = False

@dataclass
class SystemStats:
    """Global system-wide metrics tracked by the simulation engine."""
    global_time_step: int         = 0
    total_requests: int           = 0
    active_spikes: int            = 0
    sla_violations: int           = 0
    total_scale_ups: int          = 0
    total_scale_downs: int        = 0
    uptime_seconds: float         = 0.0
    system_healthy: bool          = True

@dataclass
class CloudObservation:
    """
    What the autoscaling agent observes at each time step.
    Structured to be easily converted to PyTorch tensors or LLM text prompts.
    """
    agent_id: str                 # e.g., 'frontend_controller'
    managed_service: str          # Service this agent is responsible for
    metrics: ServiceMetrics
    global_stats: SystemStats
    active_alerts: List[TrafficAlert]
    recent_history_rps: List[float] = field(default_factory=list)
    
    # RL/Agent specifics
    llm_prompt: str               = ""  # Auto-generated text representation of state
    reward: float                 = 0.0
    done: bool                    = False
    time_step: int                = 0
    max_steps: int                = 200
    legal_actions: List[str]      = field(default_factory=lambda: [
        ActionType.SCALE_UP.value, 
        ActionType.SCALE_DOWN.value, 
        ActionType.NO_OP.value
    ])

@dataclass
class ScalingAction:
    """Action issued by the RL or LLM agent to the environment."""
    agent_id: str
    action_type: ActionType       = ActionType.NO_OP
    target_service: str           = ""
    reason: str                   = ""  # Vital for LLM Chain-of-Thought logs
    raw_json: str                 = ""  # Original string output from LLM (if applicable)

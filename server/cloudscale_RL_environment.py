"""
cloud_env.py — OpenEnv-compatible environment for Cloud Autoscaling.

Wraps the CloudWorkloadGenerator and SystemDynamics into standard reset/step/state interfaces
so the autoscaling agents (RL or LLM) can be trained to handle traffic spikes.
Includes strict grading and 3 difficulty tasks for OpenEnv compliance.
"""
from __future__ import annotations

import time
import uuid
import random
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Models & Data Structures
# ---------------------------------------------------------------------------
class MicroserviceState:
    __slots__ = ("name", "min_replicas", "max_replicas", "active_pods", 
                 "pending_pods", "queue_depth", "throughput_per_pod", "metrics")
    
    def __init__(self, name: str, min_replicas: int, max_replicas: int, throughput: float):
        self.name = name
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.active_pods = min_replicas
        self.pending_pods = 0
        self.queue_depth = 0.0
        self.throughput_per_pod = throughput
        self.metrics = {"latency_p95": 0.0, "error_rate": 0.0, "rps": 0.0, "utilization": 0.0}

class CloudStats:
    def __init__(self):
        self.global_time = 0
        self.total_requests = 0
        self.sla_violations = 0
        self.active_spikes = 0
        self.start_time = time.time()
        self.system_healthy = True

class CloudStepResult:
    __slots__ = ("observations", "rewards", "done", "info")
    def __init__(self, observations: Dict[str, Any], rewards: Dict[str, Any], 
                 done: bool, info: dict):
        self.observations = observations
        self.rewards      = rewards
        self.done         = done
        self.info         = info

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class CloudAutoScalerEnv:
    """
    OpenEnv-style environment for Multi-Agent Cloud Autoscaling.
    
    Reward Shaping (Strict Discrete Buckets):
    - r_spike: 1.0 (fast scale-up), 0.5 (delayed), 0.0 (missed)
    - r_latency: 1.0 (<100ms), 0.5 (<300ms), 0.0 (>=300ms)
    - Penalty: 0.0 total reward if SLA is severely violated.
    """
    def __init__(
        self,
        task_level: str = "medium",  # "easy", "medium", or "hard"
        max_steps: int = 200,
    ) -> None:
        self.task_level = task_level.lower()
        self.max_steps = max_steps
        
        # Configure difficulty based on task
        if self.task_level == "easy":
            self.base_load = 500
            self.spike_prob = 0.0      # Predictable traffic
            self.pod_startup_delay = 1 # Instant scaling
        elif self.task_level == "hard":
            self.base_load = 800
            self.spike_prob = 0.10     # Frequent flash spikes
            self.pod_startup_delay = 5 # Severe cloud boot delays (requires anticipation)
        else: # medium
            self.base_load = 150
            self.spike_prob = 0.05     # Standard spikes
            self.pod_startup_delay = 3 # Standard delay

        # Reward Weights
        self.weights = {
            "w_latency": 0.3, "w_cost": 0.2, "w_action": 0.1, 
            "w_stability": 0.1, "w_sla": 0.1, "w_spike": 0.2
        }

        self.services: Dict[str, MicroserviceState] = {}
        self.stats = CloudStats()
        self._episode_id = ""
        self._event_queue: List[Dict] = []  # Handles delayed scaling actions
        self._historical_rps: List[float] = []
        self._spike_active = False
        self._current_spike_percentage = 0.0

    # ------------------------------------------------------------------
    # OpenEnv Interface
    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, Any]:
        self.stats = CloudStats()
        self._episode_id = str(uuid.uuid4())[:8]
        self._event_queue.clear()
        self._historical_rps = [self.base_load] * 5
        self._spike_active = False
        self._current_spike_percentage = 0.0
        
        # Initialize microservices with baseline capacities
        self.services = {
            "frontend": MicroserviceState("frontend", min_replicas=2, max_replicas=20, throughput=100),
            "backend": MicroserviceState("backend", min_replicas=2, max_replicas=50, throughput=50),
            "worker": MicroserviceState("worker", min_replicas=1, max_replicas=100, throughput=20)
        }
        return self.get_global_state()

    def step(self, actions_dict: Dict[str, str]) -> CloudStepResult:
        self.stats.global_time += 1
        
        # 1. Process Event Queue (Delayed Scaling)
        self._process_events()

        # 2. Queue New Actions & Apply Constraints
        for agent_id, action in actions_dict.items():
            if agent_id in self.services:
                self._apply_scaling_action(agent_id, action)

        # 3. Simulate Workload & Spikes
        current_rps = self._generate_workload()
        self.stats.total_requests += current_rps

        # 4. Update System Dynamics (Queuing & Latency)
        self._update_system_dynamics(current_rps)

        # 5. Compute Rewards per Agent
        rewards = {}
        for agent_id, action in actions_dict.items():
            if agent_id in self.services:
                rewards[agent_id] = self._calculate_agent_reward(agent_id, action)

        done = self.stats.global_time >= self.max_steps
        obs = self.get_global_state()

        return CloudStepResult(
            observations=obs,
            rewards=rewards,
            done=done,
            info={"active_spikes": self.stats.active_spikes, "time_step": self.stats.global_time}
        )

    def get_global_state(self) -> Dict[str, Any]:
        """Returns the full multi-agent observation space."""
        obs = {}
        for name, svc in self.services.items():
            obs[name] = {
                "active_pods": svc.active_pods / svc.max_replicas, # Normalized
                "pending_pods": svc.pending_pods / svc.max_replicas,
                "rps": svc.metrics["rps"],
                "latency_p95": svc.metrics["latency_p95"],
                "error_rate": svc.metrics["error_rate"],
                "queue_depth": svc.queue_depth,
                "utilization": svc.metrics["utilization"],
                "spike_detected": 1.0 if self._spike_active else 0.0,
                "spike_percentage": self._current_spike_percentage if self._spike_active else 0.0
            }
        return obs

    def state(self) -> Dict[str, Any]:
        """OpenEnv standard alias for getting the current state."""
        return self.get_global_state()

    def inject_spike(self, spike_percentage: float | None = None) -> float:
        """Force-activates a spike and returns the selected spike percentage."""
        if spike_percentage is None:
            spike_percentage = random.uniform(0.0, 500.0)

        self._current_spike_percentage = max(0.0, min(500.0, float(spike_percentage)))
        self._spike_active = True
        self.stats.active_spikes += 1
        return self._current_spike_percentage

    # ------------------------------------------------------------------
    # Internal Simulation Mechanics
    # ------------------------------------------------------------------
    def _process_events(self):
        """Resolves asynchronous scaling delays."""
        remaining_events = []
        for event in self._event_queue:
            if event["ready_time"] <= self.stats.global_time:
                svc = self.services[event["service"]]
                if event["type"] == "SCALE_UP" and svc.active_pods < svc.max_replicas:
                    svc.active_pods += 1
                    svc.pending_pods -= 1
            else:
                remaining_events.append(event)
        self._event_queue = remaining_events

    def _apply_scaling_action(self, service_name: str, action: str):
        svc = self.services[service_name]
        if action == "SCALE_UP":
            # Action is registered immediately, but takes effect later
            if svc.active_pods + svc.pending_pods < svc.max_replicas:
                svc.pending_pods += 1
                self._event_queue.append({
                    "time_issued": self.stats.global_time,
                    "ready_time": self.stats.global_time + self.pod_startup_delay,
                    "service": service_name,
                    "type": "SCALE_UP"
                })
        elif action == "SCALE_DOWN":
            # Scale down is usually faster/instant in Kubernetes
            if svc.active_pods > svc.min_replicas:
                svc.active_pods -= 1

    def _generate_workload(self) -> float:
        """Determines current traffic, factoring in flash spikes."""
        base = self.base_load
        
        # Simple spike logic for demonstration
        if not self._spike_active and (uuid.uuid4().int % 100) < (self.spike_prob * 100):
            self._spike_active = True
            self._current_spike_percentage = random.uniform(0.0, 500.0)
            self.stats.active_spikes += 1
            
        if self._spike_active:
            current_rps = base * (self._current_spike_percentage / 100.0)
            # Random chance to end spike
            if (uuid.uuid4().int % 100) < 15: 
                self._spike_active = False
                self._current_spike_percentage = 0.0
        else:
            # Normal noisy diurnal load
            current_rps = base + (uuid.uuid4().int % 50) - 25

        self._historical_rps.append(current_rps)
        self._historical_rps.pop(0)
        return current_rps

    def _update_system_dynamics(self, rps: float):
        """Calculates utilization, queue buildup, and latency metrics."""
        for svc in self.services.values():
            capacity = svc.active_pods * svc.throughput_per_pod
            svc.metrics["rps"] = rps
            
            if rps > capacity:
                # Queue builds up
                excess = rps - capacity
                svc.queue_depth += excess
                svc.metrics["utilization"] = 1.0
            else:
                # Process queue if capacity allows
                available = capacity - rps
                processed_from_queue = min(available, svc.queue_depth)
                svc.queue_depth -= processed_from_queue
                svc.metrics["utilization"] = rps / capacity if capacity > 0 else 1.0

            # Calculate Latency (Base + Queuing Delay penalty)
            base_latency = 20.0
            queue_penalty = (svc.queue_depth / svc.throughput_per_pod) * 10.0
            svc.metrics["latency_p95"] = base_latency + queue_penalty
            
            # Error rates spike if queue gets hopelessly large
            if svc.queue_depth > (capacity * 5):
                svc.metrics["error_rate"] = 0.10
                self.stats.sla_violations += 1
            else:
                svc.metrics["error_rate"] = 0.0

    def _calculate_agent_reward(self, agent_id: str, action: str) -> Dict[str, float]:
        """Implements the STRICT discrete bucketing reward system."""
        svc = self.services[agent_id]
        
        # 1. Penalty Override
        if svc.metrics["error_rate"] > 0.05 or svc.metrics["latency_p95"] > 500:
            return {"total": 0.0, "r_latency": 0.0, "r_cost": 0.0, "r_action": 0.0, "r_spike": 0.0, "r_sla": 0.0}

        # 2. Latency Bucket
        lat = svc.metrics["latency_p95"]
        r_latency = 1.0 if lat < 100 else (0.5 if lat < 300 else 0.0)

        # 3. Cost Bucket
        util = svc.metrics["utilization"]
        r_cost = 1.0 if 0.6 <= util <= 0.85 else (0.5 if util < 0.6 else 0.0)

        # 4. Action & Spike Handling
        r_action = 1.0
        r_spike = 1.0
        
        if self._spike_active:
            if action == "SCALE_UP":
                r_spike = 1.0 # Optimal
            elif action == "NO_OP" and svc.pending_pods > 0:
                r_spike = 0.5 # Acceptable, already scaling
            else:
                r_spike = 0.0 # Poor, ignoring the spike
                r_action = 0.0 # Wrong action for environment state
        else:
            if action == "NO_OP" and 0.4 < util < 0.8:
                r_action = 1.0 # Good stability
            elif action == "SCALE_UP" and util < 0.5:
                r_action = 0.0 # Wasting resources

        r_sla = 1.0 # Since we passed the penalty override

        # 5. Weighted Aggregation
        total = (
            self.weights["w_latency"] * r_latency +
            self.weights["w_cost"] * r_cost +
            self.weights["w_action"] * r_action +
            self.weights["w_spike"] * r_spike +
            self.weights["w_sla"] * r_sla
        )

        return {
            "total": total,
            "r_latency": r_latency,
            "r_cost": r_cost,
            "r_action": r_action,
            "r_spike": r_spike,
            "r_sla": r_sla
        }

    def grade_task(self) -> float:
        """
        Calculates the final episode score between 0.0 and 1.0.
        Balances strict SLA compliance with cost efficiency (utilization).
        """
        # If no requests were processed, score is 0
        if self.stats.total_requests == 0:
            return 0.0

        # 1. Calculate SLA Score (0.0 to 1.0)
        # Any SLA violation is penalized heavily. 
        # e.g., 5 violations out of 200 steps = 0.75 score. >20 violations = 0.0
        sla_violation_rate = min(self.stats.sla_violations / (self.max_steps * 0.1), 1.0)
        sla_score = 1.0 - sla_violation_rate

        # 2. Calculate Efficiency Score (0.0 to 1.0)
        # Did the agent waste money by over-provisioning?
        utilizations = [svc.metrics.get("utilization", 0.5) for svc in self.services.values()]
        avg_utilization = sum(utilizations) / len(utilizations) if utilizations else 0.5
        
        # Perfect utilization is around 0.75. Too low = wasted money. Too high = risky.
        if avg_utilization < 0.3:
            efficiency_score = 0.4  # Wasted a lot of money
        elif avg_utilization > 0.9:
            efficiency_score = 0.6  # Ran way too hot
        else:
            efficiency_score = 1.0  # Optimal cloud spend

        # Weighted final score: SLA is more important (70%) than cost (30%)
        final_score = (sla_score * 0.7) + (efficiency_score * 0.3)
        
        # Ensure strict 0.0 to 1.0 bounds
        return max(0.0, min(1.0, final_score))

    def close(self) -> None:
        pass
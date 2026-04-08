"""
autoscaler_agent.py — LLM-powered autonomous scaling agent.

The AutoScalerAgent receives a CloudObservation (current RPS, latency, queue depth)
and decides what action to take for its specific microservice:
  SCALE_UP   — request a new pod to handle incoming load
  SCALE_DOWN — remove a pod to save costs during low traffic
  NO_OP      — maintain current capacity

The agent can work in two modes:
  1. LLM mode   — calls an Anthropic or local Ollama model for Chain-of-Thought reasoning
  2. Rule mode  — deterministic threshold-based fallback (HPA-style)

Usage
-----
    from autoscaler_agent import AutoScalerAgent
    agent = AutoScalerAgent(managed_service="frontend", api_key="sk-ant-...")
    action = agent.decide(observation)
"""
from __future__ import annotations

import json
import os
import logging
from typing import Optional

from models import (
    CloudObservation,
    ScalingAction,
    ActionType,
    TrafficPattern
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an autonomous Site Reliability Engineering (SRE) agent managing the '{service_name}' microservice.
Your goal is to maintain strict SLA constraints (P95 Latency < 100ms, Error Rate 0%) while minimizing unnecessary infrastructure costs.
You must anticipate traffic spikes (Flash Traffic) and scale proactively due to pod startup delays.

Legal actions:
  - SCALE_UP   : Provision an additional pod. Use when queues build up or flash spikes are detected.
  - SCALE_DOWN : Terminate a pod. Use when CPU utilization is safely below 50% and traffic is stable.
  - NO_OP      : Take no action. Use when the system is stable and perfectly provisioned.

Response format (Strict JSON only, no markdown, no conversational text):
{{
  "action_type": "SCALE_UP" | "SCALE_DOWN" | "NO_OP",
  "reason": "<A concise chain-of-thought explanation for your decision>"
}}"""

USER_TEMPLATE = """Current state of '{service_name}' (Step {time_step} of {max_steps}):

SERVICE METRICS:
- Active Pods : {active_pods} / {max_replicas} (Pending: {pending_pods})
- CPU Util.   : {utilization:.1f}%
- Queue Depth : {queue_depth:.1f}
- P95 Latency : {latency:.1f}ms
- Error Rate  : {error_rate:.2f}

TRAFFIC CONDITIONS:
- Current RPS : {rps:.1f}
- Spikes?     : {alerts_text}

GLOBAL STATS:
- Uptime             : {uptime}s
- Active Flash Spikes: {active_spikes}
- SLA Violations     : {sla_violations}

Based on these metrics, what is your scaling decision?"""

# ---------------------------------------------------------------------------
# AutoScalerAgent
# ---------------------------------------------------------------------------
class AutoScalerAgent:
    """
    Autonomous agent that controls exactly one microservice.
    """
    def __init__(
        self,
        managed_service: str,
        api_key: Optional[str] = None,
        model: str = "claude-3-haiku-20240307",
        use_ollama: bool = False,
    ) -> None:
        self.managed_service = managed_service
        self.model = model
        self.use_ollama = use_ollama
        self._client = None
        
        if not self.use_ollama:
            key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            if key:
                try:
                    import anthropic
                    self._client = anthropic.Anthropic(api_key=key)
                    logger.info("AutoScalerAgent [%s]: LLM mode enabled (%s)", managed_service, model)
                except ImportError:
                    logger.warning("anthropic package missing — using rule mode")
            else:
                logger.info("AutoScalerAgent [%s]: No API key — using rule mode", managed_service)
        else:
            logger.info("AutoScalerAgent [%s]: Ollama mode selected", managed_service)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def decide(self, obs: CloudObservation) -> ScalingAction:
        """
        Given the global observation, extracts its specific service data and decides.
        Tries LLM first; falls back to rule-based engine instantly on failure.
        """
        # Ensure the observation contains data for this agent's service
        if self.managed_service not in obs:
            return ScalingAction(agent_id=self.managed_service, action_type=ActionType.NO_OP, reason="No data")

        service_data = obs[self.managed_service]

        try:
            if self._client:
                return self._llm_decide(service_data, obs)
            elif self.use_ollama:
                return self._ollama_decide(service_data, obs)
        except Exception as exc:
            logger.warning("[%s] LLM failure (%s) — falling back to deterministic rules", self.managed_service, exc)
            
        return self._rule_decide(service_data, obs)

    # ------------------------------------------------------------------
    # LLM Integrations
    # ------------------------------------------------------------------
    def _build_prompt(self, svc: dict, global_obs: dict) -> str:
        # Check if any global alerts affect this service
        active_spikes = [a for a in global_obs.get("active_alerts", []) if a["pattern"] == TrafficPattern.FLASH_SPIKE.value]
        if active_spikes:
            alerts_text = f"YES! Intensity multiplier: {active_spikes[0]['intensity_multiplier']:.1f}x"
        else:
            alerts_text = "Stable. No flash traffic detected."

        stats = global_obs.get("global_stats", {})
        
        return USER_TEMPLATE.format(
            service_name=self.managed_service,
            time_step=global_obs.get("time_step", 0),
            max_steps=global_obs.get("max_steps", 200),
            active_pods=svc.get("active_pods", 0),
            max_replicas=svc.get("max_replicas", 10),
            pending_pods=svc.get("pending_pods", 0),
            utilization=svc.get("utilization", 0.0) * 100,
            queue_depth=svc.get("queue_depth", 0.0),
            latency=svc.get("latency_p95", 0.0),
            error_rate=svc.get("error_rate", 0.0),
            rps=svc.get("rps", 0.0),
            alerts_text=alerts_text,
            uptime=stats.get("uptime_seconds", 0),
            active_spikes=stats.get("active_spikes", 0),
            sla_violations=stats.get("sla_violations", 0)
        )

    def _llm_decide(self, svc: dict, global_obs: dict) -> ScalingAction:
        user_msg = self._build_prompt(svc, global_obs)
        sys_msg = SYSTEM_PROMPT.format(service_name=self.managed_service)
        
        response = self._client.messages.create(
            model=self.model,
            max_tokens=256,
            system=sys_msg,
            messages=[{"role": "user", "content": user_msg}],
        )
        return self._parse_llm_response(response.content[0].text.strip())

    def _ollama_decide(self, svc: dict, global_obs: dict) -> ScalingAction:
        import requests
        user_msg = self._build_prompt(svc, global_obs)
        sys_msg = SYSTEM_PROMPT.format(service_name=self.managed_service)
        
        payload = {
            "model": self.model, # e.g., "llama3" or "mistral"
            "prompt": f"{sys_msg}\n\n{user_msg}",
            "stream": False,
            "format": "json"
        }
        
        res = requests.post("http://localhost:11434/api/generate", json=payload, timeout=5)
        res.raise_for_status()
        return self._parse_llm_response(res.json().get("response", ""))

    def _parse_llm_response(self, raw: str) -> ScalingAction:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
            
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("[%s] JSON parse error; defaulting to NO_OP. Raw: %s", self.managed_service, raw[:100])
            return ScalingAction(agent_id=self.managed_service, action_type=ActionType.NO_OP, reason="JSON parse failure")

        # Map string to Enum securely
        act_str = data.get("action_type", "NO_OP")
        action_type = getattr(ActionType, act_str, ActionType.NO_OP)

        return ScalingAction(
            agent_id=self.managed_service,
            action_type=action_type,
            reason=data.get("reason", "LLM decided without specific reason."),
            raw_json=raw
        )

    # ------------------------------------------------------------------
    # Deterministic Baseline (HPA logic)
    # ------------------------------------------------------------------
    def _rule_decide(self, svc: dict, global_obs: dict) -> ScalingAction:
        """
        Acts exactly like a standard Kubernetes Horizontal Pod Autoscaler.
        Provides a baseline to compare your LLM/RL agent against.
        """
        latency = svc.get("latency_p95", 0.0)
        queue = svc.get("queue_depth", 0.0)
        utilization = svc.get("utilization", 0.0)
        pending = svc.get("pending_pods", 0)
        
        active_spikes = [a for a in global_obs.get("active_alerts", []) if a["pattern"] == TrafficPattern.FLASH_SPIKE.value]

        # 1. Proactive Spike Handling
        if active_spikes and pending == 0:
            return ScalingAction(self.managed_service, ActionType.SCALE_UP, reason="Rule Engine: Flash spike detected.")

        # 2. Reactive Thresholds
        if (utilization > 0.85 or queue > 5.0 or latency > 150) and pending == 0:
            return ScalingAction(self.managed_service, ActionType.SCALE_UP, reason="Rule Engine: Utilization/Latency critical.")
            
        # 3. Cost Saving
        if utilization < 0.40 and queue == 0 and not active_spikes:
            return ScalingAction(self.managed_service, ActionType.SCALE_DOWN, reason="Rule Engine: Low utilization, saving costs.")

        return ScalingAction(self.managed_service, ActionType.NO_OP, reason="Rule Engine: System stable.")

# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
def create_agent_swarm(services: list[str], use_ollama: bool = False) -> dict[str, AutoScalerAgent]:
    """Instantiates a multi-agent topology."""
    return {
        svc: AutoScalerAgent(managed_service=svc, use_ollama=use_ollama)
        for svc in services
    }

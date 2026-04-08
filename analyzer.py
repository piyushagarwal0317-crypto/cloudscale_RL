"""
analyzer.py — Traffic analysis and alerting engine for the Cloud Autoscaler.

Consumes real-time microservice metrics and RPS histories to produce:
  - Microservice HealthStatus (HEALTHY, DEGRADED, CRITICAL)
  - TrafficAlerts (Flash Spike detection)
  - A composite System Stress Score for the dashboard
"""
from __future__ import annotations

import time
import statistics
from collections import deque
from typing import Dict, List, Tuple, Optional

from models import (
    ServiceMetrics,
    TrafficAlert,
    TrafficPattern,
    HealthStatus
)

# ---------------------------------------------------------------------------
# Thresholds (Tuning dials for the hackathon simulation)
# ---------------------------------------------------------------------------
SPIKE_MULTIPLIER_THRESHOLD = 2.5   # RPS > 2.5x the rolling average = Flash Spike
LATENCY_DEGRADED_MS        = 100.0 # P95 > 100ms
LATENCY_CRITICAL_MS        = 300.0 # P95 > 300ms
QUEUE_DANGER_RATIO         = 1.5   # Queue depth > 1.5x total service capacity
ERROR_RATE_TOLERANCE       = 0.02  # 2% error rate triggers critical alerts

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default

# ---------------------------------------------------------------------------
# TrafficAnalyzer
# ---------------------------------------------------------------------------
class TrafficAnalyzer:
    """
    Stateless engine that evaluates raw infrastructure metrics.
    Turns numbers into actionable states (HealthStatus) and Alerts.
    """
    def __init__(self) -> None:
        self._alert_counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(
        self,
        services: Dict[str, ServiceMetrics],
        rps_history: List[float],
        existing_alerts: Optional[List[TrafficAlert]] = None,
    ) -> Tuple[Dict[str, HealthStatus], List[TrafficAlert], float]:
        """
        Main entry point. Evaluates the current state of the system.
        
        Returns
        -------
        health_states : dict[service_name -> HealthStatus]
        new_alerts : list[TrafficAlert]
        system_stress_score : float [0.0 to 1.0]
        """
        if not services:
            return {}, [], 0.0

        new_alerts: List[TrafficAlert] = []
        health_states: Dict[str, HealthStatus] = {}
        existing_ids = {al.alert_id for al in (existing_alerts or [])}

        # 1. Global Traffic Spike Detection
        current_rps = rps_history[-1] if rps_history else 0.0
        is_spike, multiplier = self._detect_flash_spike(current_rps, rps_history)
        
        if is_spike:
            # Generate an alert for the frontend (the entrypoint)
            alert = self._create_alert(
                service_id="global_ingress",
                pattern=TrafficPattern.FLASH_SPIKE,
                intensity=multiplier,
                desc=f"Sudden traffic surge detected: {multiplier:.1f}x baseline load."
            )
            new_alerts.append(alert)

        # 2. Per-Service Health Assessment
        total_stress = 0.0
        for name, metrics in services.items():
            health, stress = self._assess_service_health(metrics)
            health_states[name] = health
            total_stress += stress
            
            # Generate critical alerts if a service is drowning
            if health == HealthStatus.CRITICAL:
                new_alerts.append(self._create_alert(
                    service_id=name,
                    pattern=TrafficPattern.DECAY,
                    intensity=1.0,
                    desc=f"Service '{name}' is CRITICAL. P95: {metrics.latency_p95:.1f}ms, Queue: {metrics.queue_depth:.1f}"
                ))

        # Normalize system stress score between 0 and 1
        system_stress_score = min(1.0, _safe_div(total_stress, len(services)))

        return health_states, new_alerts, system_stress_score

    # ------------------------------------------------------------------
    # Analysis Mechanics
    # ------------------------------------------------------------------
    def _detect_flash_spike(self, current_rps: float, history: List[float]) -> Tuple[bool, float]:
        """
        Calculates if the current RPS is a mathematically significant anomaly 
        compared to the recent baseline.
        """
        if len(history) < 3:
            return False, 1.0
            
        # Baseline is the average of the history, excluding the current tick
        baseline_rps = statistics.mean(history[:-1])
        if baseline_rps < 10:  # Prevent division by zero / noise on startup
            return False, 1.0
            
        multiplier = current_rps / baseline_rps
        is_spike = multiplier >= SPIKE_MULTIPLIER_THRESHOLD
        
        return is_spike, multiplier

    def _assess_service_health(self, metrics: ServiceMetrics) -> Tuple[HealthStatus, float]:
        """
        Evaluates queuing theory metrics to assign a strict Health Status.
        Returns: (HealthStatus, service_stress_score)
        
        Stress Score Formula:
        $$Stress = (0.5 \\cdot \\text{utilization}) + (0.5 \\cdot \\min(1.0, \\frac{\\text{latency}}{300}))$$
        """
        stress_score = (0.5 * metrics.utilization) + (0.5 * min(1.0, _safe_div(metrics.latency_p95, LATENCY_CRITICAL_MS)))
        
        # Hard constraints for CRITICAL
        if metrics.latency_p95 >= LATENCY_CRITICAL_MS or metrics.error_rate >= ERROR_RATE_TOLERANCE:
            return HealthStatus.CRITICAL, min(1.0, stress_score + 0.3)
            
        # Constraints for DEGRADED
        capacity = metrics.active_pods * metrics.throughput_per_pod
        queue_ratio = _safe_div(metrics.queue_depth, capacity)
        
        if metrics.latency_p95 >= LATENCY_DEGRADED_MS or queue_ratio > QUEUE_DANGER_RATIO:
            return HealthStatus.DEGRADED, min(1.0, stress_score + 0.1)
            
        return HealthStatus.HEALTHY, min(1.0, stress_score)

    def _create_alert(self, service_id: str, pattern: TrafficPattern, intensity: float, desc: str) -> TrafficAlert:
        self._alert_counter += 1
        return TrafficAlert(
            alert_id=f"alert_spk_{self._alert_counter:04d}",
            timestamp=time.time(),
            service_id=service_id,
            pattern=pattern,
            intensity_multiplier=intensity,
            description=desc,
            acknowledged=False
        )

# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------
def analyze_traffic(
    services: Dict[str, ServiceMetrics],
    rps_history: List[float],
    existing_alerts: Optional[List[TrafficAlert]] = None,
) -> Tuple[Dict[str, HealthStatus], List[TrafficAlert], float]:
    """Module-level wrapper — creates a fresh TrafficAnalyzer and runs analysis."""
    return TrafficAnalyzer().analyze(services, rps_history, existing_alerts)

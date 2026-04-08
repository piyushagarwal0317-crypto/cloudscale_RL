"""
workload_generator.py — Traffic Simulation Engine
=================================================
Generates realistic cloud infrastructure workloads including diurnal (day/night) 
patterns, stochastic noise, and sudden flash spikes.

This acts as the heartbeat for the CloudAutoScalerEnv.
"""

import math
import time
import random
from enum import Enum
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Internal Enums & Dataclasses 
# (Kept local to prevent polluting the OpenEnv Pydantic models)
# ---------------------------------------------------------------------------
class TrafficPattern(str, Enum):
    NORMAL      = "normal"
    FLASH_SPIKE = "flash_spike"    
    DIURNAL     = "diurnal"        
    DECAY       = "decay"          

@dataclass
class TrafficAlert:
    alert_id: str
    timestamp: float
    service_id: str
    pattern: TrafficPattern
    intensity_multiplier: float    
    description: str
    acknowledged: bool = False

@dataclass
class SystemStats:
    global_time_step: int = 0
    total_requests: int = 0
    active_spikes: int = 0
    sla_violations: int = 0
    total_scale_ups: int = 0
    total_scale_downs: int = 0
    uptime_seconds: float = 0.0
    system_healthy: bool = True

# ---------------------------------------------------------------------------
# Workload Generator Configuration
# ---------------------------------------------------------------------------
class WorkloadConfig:
    def __init__(self, base_load: float = 500.0, diurnal_amplitude: float = 200.0, spike_prob: float = 0.05):
        self.base_load = base_load
        self.diurnal_amplitude = diurnal_amplitude
        self.spike_prob = spike_prob
        
        # How many ticks a full "day" takes
        self.diurnal_period_ticks = 100 
        
        # Max severity of a flash spike (multiplier over current load)
        self.max_spike_multiplier = 5.0 

# ---------------------------------------------------------------------------
# Core Workload Engine
# ---------------------------------------------------------------------------
class CloudWorkloadGenerator:
    """
    Simulates realistic web traffic. 
    Stateful — maintains current spike state and time step.
    """
    def __init__(self, config: WorkloadConfig = None):
        self.config = config or WorkloadConfig()
        self._time_step = 0
        
        self._active_spike = False
        self._spike_multiplier = 1.0
        self._spike_duration_remaining = 0
        
    def step(self) -> Tuple[float, TrafficPattern]:
        """
        Advances the simulation by one tick.
        Returns: (current_rps, active_pattern)
        """
        self._time_step += 1
        
        # 1. Base Diurnal Load (Sine wave)
        # Calculates where we are in the "day"
        progress = (self._time_step % self.config.diurnal_period_ticks) / self.config.diurnal_period_ticks
        diurnal_wave = math.sin(progress * 2 * math.pi) * self.config.diurnal_amplitude
        
        # 2. Add Stochastic Noise (Jitter)
        noise = random.uniform(-50.0, 50.0)
        
        current_rps = self.config.base_load + diurnal_wave + noise
        
        # 3. Handle Flash Spikes
        pattern = TrafficPattern.DIURNAL
        
        if self._active_spike:
            self._spike_duration_remaining -= 1
            if self._spike_duration_remaining <= 0:
                self._active_spike = False
                self._spike_multiplier = 1.0
                pattern = TrafficPattern.DECAY
            else:
                current_rps *= self._spike_multiplier
                pattern = TrafficPattern.FLASH_SPIKE
        else:
            # Check if a new spike should start
            if random.random() < self.config.spike_prob:
                self._active_spike = True
                self._spike_multiplier = random.uniform(2.5, self.config.max_spike_multiplier)
                self._spike_duration_remaining = random.randint(3, 10) # Spike lasts 3-10 ticks
                current_rps *= self._spike_multiplier
                pattern = TrafficPattern.FLASH_SPIKE
                
        # Prevent negative RPS
        current_rps = max(10.0, current_rps)
        
        return current_rps, pattern
        
    def force_spike(self, multiplier: float = 4.0, duration: int = 5):
        """Manual override for testing/chaos engineering."""
        self._active_spike = True
        self._spike_multiplier = multiplier
        self._spike_duration_remaining = duration

# ---------------------------------------------------------------------------
# State Tracker
# ---------------------------------------------------------------------------
class SystemStateTracker:
    """
    Maintains the history and rolling stats of the environment.
    This tracks the global metrics across all microservices.
    """
    def __init__(self):
        self.stats = SystemStats()
        self.rps_history: List[float] = []
        self.service_metrics: Dict[str, Any] = {}
        self.active_alerts: List[TrafficAlert] = []
        
    def update(self, current_rps: float, services: Dict[str, Any]):
        """Called every tick by the CloudAutoScalerEnv."""
        self.stats.global_time_step += 1
        self.stats.total_requests += int(current_rps)
        
        self.rps_history.append(current_rps)
        # Keep only the last 100 ticks to save memory
        if len(self.rps_history) > 100:
            self.rps_history.pop(0)
            
        self.service_metrics = services
        
    def add_alerts(self, alerts: List[TrafficAlert]):
        """Adds new alerts and increments alert stats."""
        for alert in alerts:
            self.active_alerts.append(alert)
            if alert.pattern == TrafficPattern.FLASH_SPIKE:
                self.stats.active_spikes += 1
                
        # Keep only recent alerts
        if len(self.active_alerts) > 50:
            self.active_alerts = self.active_alerts[-50:]
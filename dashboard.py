"""
app_dashboard.py — Cloud Ops & Autoscaling Dashboard

A Grafana-inspired Gradio dashboard for monitoring the CloudAutoScalerEnv.
Three tabs:
----------
Tab 1 — Cluster Status (Live microservice cards, active pods, health)
Tab 2 — Telemetry (RPS vs Time, Latency vs Time charts)
Tab 3 — Chaos & Logs (Manual spike injection, scaling action logs)
"""
from __future__ import annotations

import os
import time
import threading
import logging
from collections import deque
from typing import Dict, List, Tuple

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Import your new cloud models
from server.cloudscale_RL_environment import CloudAutoScalerEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration & State Simulation
# ---------------------------------------------------------------------------
REFRESH_SEC = int(os.environ.get("REFRESH_SEC", "2"))

# Initialize the local environment
_env = CloudAutoScalerEnv(task_level="medium")
_env.reset()

# Telemetry History for Charts (Storing last 60 ticks)
_time_history = deque(maxlen=60)
_rps_history = deque(maxlen=60)
_latency_history = {"frontend": deque(maxlen=60), "backend": deque(maxlen=60), "worker": deque(maxlen=60)}

def _autoscale_actions(obs: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """Simple heuristic autoscaler so pods and utilization react to spike intensity."""
    actions: Dict[str, str] = {}
    spike_pct = float(obs.get("frontend", {}).get("spike_percentage", 0.0))

    for svc_name, metrics in obs.items():
        svc = _env.services[svc_name]
        util = float(metrics.get("utilization", 0.0))
        queue_depth = float(metrics.get("queue_depth", 0.0))

        if (spike_pct >= 100.0 or util > 0.85 or queue_depth > 10.0) and (svc.active_pods + svc.pending_pods < svc.max_replicas):
            actions[svc_name] = "SCALE_UP"
        elif spike_pct < 25.0 and util < 0.35 and queue_depth < 1.0 and svc.active_pods > svc.min_replicas:
            actions[svc_name] = "SCALE_DOWN"
        else:
            actions[svc_name] = "NO_OP"

    return actions

def _simulation_loop():
    """Background thread: Steps the environment automatically to generate live data."""
    while True:
        time.sleep(1.0) # 1 tick per second for live demo purposes

        # Apply a simple autoscaling controller so pods react to spike intensity and utilization.
        current_obs = _env.get_global_state()
        actions = _autoscale_actions(current_obs)
        step_result = _env.step(actions)
        obs = step_result.observations
        done = step_result.done
        
        # Record Telemetry
        _time_history.append(_env.stats.global_time)
        _rps_history.append(obs["frontend"]["rps"]) # Frontend RPS represents global ingress
        for svc in ["frontend", "backend", "worker"]:
            _latency_history[svc].append(obs[svc]["latency_p95"])
            
        if done:
            _env.reset()

# Start background simulation
threading.Thread(target=_simulation_loop, daemon=True).start()

# ---------------------------------------------------------------------------
# "Grafana" Cyberpunk Styling & Logic
# ---------------------------------------------------------------------------
HEALTHY = "healthy"
DEGRADED = "degraded"
CRITICAL = "critical"

_HEALTH_COLOURS = {
    HEALTHY:  "#00E676", # Neon Green
    DEGRADED: "#FFEA00", # Neon Yellow
    CRITICAL: "#FF1744", # Neon Red
}

_CHART_BG = "#0B1021"
_GRID_COLOUR = "#1A233A"
_TEXT_COLOUR = "#8B9BB4"
_CYAN = "#00E5FF"
_MAGENTA = "#F50057"

def _style_ax(ax, title=""):
    ax.set_facecolor(_CHART_BG)
    ax.tick_params(colors=_TEXT_COLOUR, labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(_GRID_COLOUR)
    ax.spines["left"].set_color(_GRID_COLOUR)
    ax.grid(True, alpha=0.3, color=_GRID_COLOUR, linestyle="--")
    if title:
        ax.set_title(title, color="#FFFFFF", fontweight="bold", fontsize=10, pad=10)

# ---------------------------------------------------------------------------
# Tab 1 — Cluster Status (The Command Center)
# ---------------------------------------------------------------------------
def _build_cluster_html() -> str:
    obs = _env.get_global_state()
    spike_pct = float(obs.get("frontend", {}).get("spike_percentage", 0.0))
    spike_label = "ACTIVE" if spike_pct > 0 else "IDLE"
    spike_color = "#F59E0B" if spike_pct > 0 else "#00E676"
    
    html = f'''
    <div style="display:flex; justify-content:center; margin-bottom:16px;">
        <div style="background:#111827; border:1px solid #1F2937; border-top:4px solid {spike_color};
                    border-radius:8px; padding:16px 24px; min-width:260px; text-align:center;
                    box-shadow:0 4px 6px rgba(0,0,0,0.3);">
            <div style="color:#9CA3AF; font-size:0.75rem; letter-spacing:1px;">SPIKES</div>
            <div style="color:{spike_color}; font-size:1.8rem; font-weight:bold; font-family:monospace;">{spike_pct:.1f}%</div>
            <div style="color:#E5E7EB; font-size:0.8rem; font-family:monospace;">{spike_label}</div>
        </div>
    </div>
    <div style="display:flex; gap:16px; flex-wrap:wrap; justify-content:center;">
    '''
    
    for svc_name, metrics in obs.items():
        # Convert normalized metrics back for display
        active_pods = int(metrics["active_pods"] * _env.services[svc_name].max_replicas)
        latency = metrics["latency_p95"]
        util = metrics["utilization"] * 100
        queue_depth = metrics["queue_depth"]
        
        # Determine Health Dynamically
        health = HEALTHY
        if latency > 100: health = DEGRADED
        if latency > 300: health = CRITICAL
        h_color = _HEALTH_COLOURS[health]
        
        card = f"""
        <div style="background:#111827; border: 1px solid #1F2937; border-top: 4px solid {h_color}; 
                    border-radius: 8px; padding: 20px; width: 30%; min-width: 250px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 12px;">
                <h3 style="color:#F3F4F6; margin:0; font-family:monospace; font-size:1.2rem; text-transform:uppercase;">
                    &#9881; {svc_name}
                </h3>
                <span style="background:{h_color}22; color:{h_color}; padding:2px 8px; border-radius:12px; font-size:0.7rem; font-weight:bold;">
                    {health.upper()}
                </span>
            </div>
            
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                <div>
                    <div style="color:#9CA3AF; font-size:0.75rem;">ACTIVE PODS</div>
                    <div style="color:#3B82F6; font-size:1.5rem; font-weight:bold; font-family:monospace;">{active_pods}</div>
                </div>
                <div>
                    <div style="color:#9CA3AF; font-size:0.75rem;">P95 LATENCY</div>
                    <div style="color:{h_color}; font-size:1.5rem; font-weight:bold; font-family:monospace;">{latency:.0f}ms</div>
                </div>
                <div>
                    <div style="color:#9CA3AF; font-size:0.75rem;">CPU UTILIZATION</div>
                    <div style="color:#F3F4F6; font-size:1.2rem; font-family:monospace;">{util:.1f}%</div>
                </div>
                <div>
                    <div style="color:#9CA3AF; font-size:0.75rem;">QUEUE DEPTH</div>
                    <div style="color:#F59E0B; font-size:1.2rem; font-family:monospace;">{queue_depth:.0f}</div>
                </div>
            </div>
        </div>
        """
        html += card
        
    # Add global spike status
    spike_active = "YES" if _env._spike_active else "NO"
    spike_factor = f"{_env._current_spike_percentage:.1f}%" if _env._spike_active else "N/A"
    spike_color = "#EF4444" if _env._spike_active else "#10B981"
    
    spike_card = f"""
    <div style="background:#111827; border: 1px solid #1F2937; border-top: 4px solid {spike_color}; 
                border-radius: 8px; padding: 20px; width: 30%; min-width: 250px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-top: 16px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 12px;">
            <h3 style="color:#F3F4F6; margin:0; font-family:monospace; font-size:1.2rem; text-transform:uppercase;">
                &#9889; SPIKE STATUS
            </h3>
            <span style="background:{spike_color}22; color:{spike_color}; padding:2px 8px; border-radius:12px; font-size:0.7rem; font-weight:bold;">
                {spike_active}
            </span>
        </div>
        
        <div style="display:grid; grid-template-columns: 1fr; gap: 12px;">
            <div>
                <div style="color:#9CA3AF; font-size:0.75rem;">CURRENT SPIKE VALUE</div>
                <div style="color:#F3F4F6; font-size:1.5rem; font-weight:bold; font-family:monospace;">{spike_factor}</div>
            </div>
        </div>
    </div>"""
    
    html += spike_card
    html += '</div>'
    return html

def refresh_cluster_tab():
    return _build_cluster_html()

# ---------------------------------------------------------------------------
# Tab 2 — Telemetry Charts
# ---------------------------------------------------------------------------
def _build_telemetry_charts():
    if not _time_history:
        # Return empty figures to prevent Gradio errors on fast reloads
        return Figure(figsize=(10, 3.5)), Figure(figsize=(10, 3.5))
        
    times = list(_time_history)
    
    # Chart 1: Global Traffic (RPS)
    fig_rps = Figure(figsize=(10, 3.5))
    ax1 = fig_rps.add_subplot(111)
    fig_rps.patch.set_facecolor(_CHART_BG)
    _style_ax(ax1, "Global Ingress Traffic (Requests Per Second)")
    
    ax1.plot(times, list(_rps_history), color=_CYAN, linewidth=2)
    ax1.fill_between(times, list(_rps_history), alpha=0.1, color=_CYAN)
    fig_rps.tight_layout()

    # Chart 2: P95 Latency by Service
    fig_lat = Figure(figsize=(10, 3.5))
    ax2 = fig_lat.add_subplot(111)
    fig_lat.patch.set_facecolor(_CHART_BG)
    _style_ax(ax2, "P95 Latency (ms) - SLA Limit: 100ms")
    
    colors = {"frontend": _CYAN, "backend": _MAGENTA, "worker": "#00E676"}
    for svc, latencies in _latency_history.items():
        if latencies:
            ax2.plot(times, list(latencies), label=svc.upper(), color=colors[svc], linewidth=2)
            
    # Draw SLA Warning Line
    ax2.axhline(100, color="#FF1744", linestyle="--", linewidth=1, alpha=0.7)
    ax2.legend(facecolor=_CHART_BG, edgecolor=_GRID_COLOUR, labelcolor=_TEXT_COLOUR, loc="upper left")
    fig_lat.tight_layout()

    return fig_rps, fig_lat

# ---------------------------------------------------------------------------
# Tab 3 — Chaos Engineering & Logs
# ---------------------------------------------------------------------------
def _trigger_flash_spike():
    """Manual override to force the workload generator to spike."""
    spike_pct = _env.inject_spike()
    return f"🔥 FLASH SPIKE INJECTED: Traffic set to {spike_pct:.1f}%"

def _reset_environment():
    _env.reset()
    _time_history.clear()
    _rps_history.clear()
    for q in _latency_history.values(): q.clear()
    return "♻️ Environment Reset. Traffic normalized."

# ---------------------------------------------------------------------------
# Gradio UI Layout
# ---------------------------------------------------------------------------
CYBER_CLOUD_CSS = """
body, .gradio-container { background: #050B14 !important; color: #E2E8F0 !important; font-family: 'Inter', sans-serif; }
.tabs > .tab-nav { background: #0B1021 !important; border-bottom: 2px solid #1A233A !important; }
.tabs > .tab-nav > button { color: #8B9BB4 !important; font-weight: bold !important; text-transform: uppercase; letter-spacing: 1px; }
.tabs > .tab-nav > button.selected { color: #00E5FF !important; border-bottom: 2px solid #00E5FF !important; }
button.primary { background: #F50057 !important; color: white !important; font-weight: bold !important; border-radius: 4px !important; border: none !important; }
button.secondary { background: #1A233A !important; color: #00E5FF !important; border: 1px solid #00E5FF !important; border-radius: 4px !important;}
"""

with gr.Blocks(title="CloudScale Autoscaler Dashboard", css=CYBER_CLOUD_CSS) as demo:
    
    gr.HTML("""
    <div style="background:#0B1021; padding:15px; border-bottom: 1px solid #1A233A; margin-bottom:20px; display:flex; justify-content:space-between; align-items:center;">
        <div>
            <h1 style="color:#FFFFFF; margin:0; font-size:1.5rem; letter-spacing:1px;">🌩️ CloudScale OpsCenter</h1>
            <span style="color:#00E5FF; font-size:0.8rem; font-family:monospace;">AUTOSCALING RL AGENT TELEMETRY</span>
        </div>
        <div style="color:#00E676; font-family:monospace; font-weight:bold;">● LIVE METRICS STREAMING</div>
    </div>
    """)

    with gr.Tabs():
        # TAB 1: CLUSTER
        with gr.TabItem("Cluster Status"):
            cluster_html = gr.HTML()

        # TAB 2: TELEMETRY
        with gr.TabItem("Telemetry & Metrics"):
            rps_chart = gr.Plot()
            lat_chart = gr.Plot()

        # TAB 3: CHAOS CONTROL
        with gr.TabItem("Chaos Engineering"):
            gr.Markdown("### ⚠️ Inject System Failures & Spikes\nTest your RL Agent's response time by manually overriding the environment dynamics.")
            with gr.Row():
                btn_spike = gr.Button("🔥 INJECT FLASH SPIKE", variant="primary")
                btn_reset = gr.Button("♻️ RESET ENVIRONMENT", variant="secondary")
            
            chaos_output = gr.Textbox(label="System Console", interactive=False)
            btn_spike.click(fn=_trigger_flash_spike, outputs=chaos_output)
            btn_reset.click(fn=_reset_environment, outputs=chaos_output)

    # Auto-Refresh Timers
    timer = gr.Timer(REFRESH_SEC)
    timer.tick(fn=refresh_cluster_tab, outputs=[cluster_html])
    timer.tick(fn=_build_telemetry_charts, outputs=[rps_chart, lat_chart])

if __name__ == "__main__":
    import sys
    share = "--share" in sys.argv
    print("🚀 Starting CloudScale Autoscaler Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:7861")
    print("🌐 If running in VS Code, it should open automatically")
    print("💡 If not, manually open: http://localhost:7861 in your browser")
    if share:
        print("🔗 Creating public link...")
    demo.launch(server_name="0.0.0.0", server_port=7861, share=share)
else:
    # Export demo for integration with FastAPI
    dashboard_app = demo
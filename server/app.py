from __future__ import annotations

import os
import time
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Import the updated environment
from server.cloudscale_RL_environment import CloudAutoScalerEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the dashboard for integration
try:
    from dashboard import dashboard_app
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    logger.warning("Dashboard not available - run from project root to enable /dashboard endpoint")

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

# Mount Gradio dashboard if available
if DASHBOARD_AVAILABLE:
    import gradio as gr
    gr.mount_gradio_app(app, dashboard_app, path="/dashboard")
    logger.info("Dashboard mounted at /dashboard")

# ---------------------------------------------------------------------------
# Web Dashboard HTML
# ---------------------------------------------------------------------------
HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>CloudScale RL Dashboard</title>
    <style>
        :root {
            color-scheme: dark;
            --bg: #0f172a;
            --card: rgba(15, 23, 42, 0.96);
            --card-border: rgba(148, 163, 184, 0.18);
            --text: #e2e8f0;
            --muted: #94a3b8;
            --primary: #7c3aed;
            --primary-strong: #a855f7;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --shadow: 0 24px 80px rgba(15, 23, 42, 0.35);
        }
        * { box-sizing: border-box; }
        html, body {
            margin: 0;
            min-height: 100%;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: radial-gradient(circle at top left, rgba(124, 58, 237, 0.18), transparent 28%),
                        radial-gradient(circle at bottom right, rgba(34, 197, 94, 0.14), transparent 22%),
                        linear-gradient(180deg, #020617 0%, #111827 100%);
            color: var(--text);
        }
        body { padding: 24px; }
        .page { max-width: 1180px; margin: 0 auto; }
        .hero {
            padding: 36px;
            border: 1px solid var(--card-border);
            border-radius: 28px;
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.9), rgba(15, 23, 42, 0.82));
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
        }
        .hero h1 {
            font-size: clamp(2.4rem, 4vw, 3.6rem);
            line-height: 1.05;
            margin-bottom: 16px;
        }
        .hero p {
            color: var(--muted);
            font-size: 1.05rem;
            max-width: 760px;
            margin-bottom: 24px;
        }
        .hero .hero-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }
        .button, .link-button {
            border: none;
            border-radius: 999px;
            padding: 14px 22px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.18s ease, background 0.18s ease, box-shadow 0.18s ease;
        }
        .button:hover, .link-button:hover { transform: translateY(-1px); }
        .button.primary { background: linear-gradient(135deg, var(--primary), var(--primary-strong)); color: white; box-shadow: 0 16px 30px rgba(124, 58, 237, 0.24); }
        .button.secondary { background: rgba(148, 163, 184, 0.12); color: var(--text); }
        .button.success { background: var(--success); color: white; }
        .button.warning { background: var(--warning); color: white; }
        .button.danger { background: var(--danger); color: white; }
        .link-button {
            background: transparent;
            color: var(--text);
            border: 1px solid rgba(148, 163, 184, 0.24);
            text-decoration: none;
        }
        .grid {
            display: grid;
            gap: 24px;
            margin-top: 28px;
        }
        .summary-grid {
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        }
        .card {
            background: var(--card);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 24px;
            box-shadow: var(--shadow);
        }
        .card h2 {
            font-size: 1.1rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 18px;
            color: var(--muted);
        }
        .metric {
            display: grid;
            gap: 10px;
            margin-bottom: 16px;
        }
        .metric strong { display: block; color: white; font-size: 1.9rem; line-height: 1.1; }
        .metric span { color: var(--muted); }
        .inline-label { color: var(--muted); font-size: 0.95rem; }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            font-size: 0.95rem;
            background: rgba(71, 85, 105, 0.18);
            color: var(--text);
        }
        .button-row {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }
        .state-card-grid {
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        }
        .agent-card {
            border-radius: 20px;
            padding: 18px;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(148, 163, 184, 0.12);
        }
        .agent-card h3 {
            margin: 0 0 12px;
            font-size: 1.05rem;
            color: white;
        }
        .agent-list {
            list-style: none;
            padding: 0;
            margin: 0;
            display: grid;
            gap: 10px;
        }
        .agent-list li {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            color: var(--muted);
            font-size: 0.95rem;
        }
        .agent-list li span:last-child { color: white; font-weight: 600; }
        .log-panel {
            min-height: 320px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        .log-output {
            flex: 1;
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 18px;
            padding: 18px;
            overflow-y: auto;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            font-size: 0.92rem;
            color: #d1d5db;
            white-space: pre-wrap;
            line-height: 1.5;
        }
        .log-actions {
            margin-top: 16px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .links-list {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 12px;
        }
        .links-list a {
            color: var(--text);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 999px;
            padding: 10px 16px;
            text-decoration: none;
            transition: background 0.18s ease;
        }
        .links-list a:hover { background: rgba(124, 58, 237, 0.16); }
        @media (max-width: 720px) {
            body { padding: 16px; }
            .hero { padding: 28px; }
        }
    </style>
</head>
<body>
    <div class="page">
        <section class="hero">
            <h1>CloudScale RL</h1>
            <p>Interactive HF Space dashboard for the CloudScale autoscaling environment. View live state, run control actions, and inspect responses without leaving the browser.</p>
            <div class="hero-actions">
                <button class="button primary" onclick="postReset()">Reset Environment</button>
                <button class="button secondary" onclick="fetchState()">Refresh State</button>
                <a class="link-button" href="/docs">API Docs</a>
                <a class="link-button" href="/redoc">Redoc</a>
            </div>
            <div class="metric" style="margin-top: 28px;">
                <strong id="summaryStateLabel">Waiting for live environment state...</strong>
                <span id="summaryStateDetail">Click "Refresh State" or perform an action to populate the dashboard.</span>
            </div>
        </section>

        <div class="grid summary-grid" style="margin-top: 24px;">
            <div class="card">
                <h2>Live Metrics</h2>
                <div class="metric">
                    <strong id="summaryAgents">—</strong>
                    <span>Tracked Agents</span>
                </div>
                <div class="metric">
                    <strong id="summaryActivePods">—</strong>
                    <span>Total Active Pods</span>
                </div>
                <div class="metric">
                    <strong id="summaryPendingPods">—</strong>
                    <span>Total Pending Pods</span>
                </div>
                <div class="metric">
                    <strong id="summaryRps">—</strong>
                    <span>Peak RPS</span>
                </div>
            </div>

            <div class="card">
                <h2>Control Panel</h2>
                <div class="button-row">
                    <button class="button success" onclick="postStep('SCALE_UP')">Scale Up</button>
                    <button class="button danger" onclick="postStep('SCALE_DOWN')">Scale Down</button>
                    <button class="button secondary" onclick="postStep('NO_OP')">No-op</button>
                </div>
                <p class="inline-label" style="margin-top: 16px;">Note: actions are sent to all managed agents with the same command.</p>
            </div>

            <div class="card">
                <h2>Quick Links</h2>
                <div class="links-list">
                    <a href="/docs">Interactive docs</a>
                    <a href="/redoc">Redoc guide</a>
                    <a href="https://huggingface.co/spaces/bitmain/cloudscale_RL" target="_blank">HF Space</a>
                    <a href="https://github.com/piyushagarwal0317-crypto/cloudscale_RL" target="_blank">GitHub</a>
                </div>
            </div>
        </div>

        <div class="grid state-card-grid" style="margin-top: 24px;">
            <div class="card">
                <h2>Agent State</h2>
                <div id="agentCards" class="grid state-card-grid">
                    <div class="agent-card">
                        <h3>Ready</h3>
                        <p class="inline-label">Use refresh or send an action to populate agent cards.</p>
                    </div>
                </div>
            </div>

            <div class="card log-panel">
                <h2>Response Log</h2>
                <div id="logOutput" class="log-output">No activity yet.</div>
                <div class="log-actions">
                    <button class="button secondary" onclick="clearLog()">Clear Log</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        const logEntries = [];

        function formatNumber(value) {
            return typeof value === 'number' ? value.toLocaleString('en-US', { maximumFractionDigits: 2 }) : value ?? '—';
        }

        function renderJson(value) {
            return JSON.stringify(value, null, 2);
        }

        function updateSummary(state) {
            const agents = Object.keys(state);
            const active = agents.reduce((sum, id) => sum + (state[id]?.active_pods || 0), 0);
            const pending = agents.reduce((sum, id) => sum + (state[id]?.pending_pods || 0), 0);
            const peakRps = agents.reduce((max, id) => Math.max(max, state[id]?.rps || 0), 0);
            document.getElementById('summaryAgents').textContent = agents.length;
            document.getElementById('summaryActivePods').textContent = formatNumber(active);
            document.getElementById('summaryPendingPods').textContent = formatNumber(pending);
            document.getElementById('summaryRps').textContent = formatNumber(peakRps);
            document.getElementById('summaryStateLabel').textContent = `Updated ${new Date().toLocaleTimeString()}`;
            document.getElementById('summaryStateDetail').textContent = `${agents.length} agents reporting live state.`;
        }

        function renderAgentCards(state) {
            const container = document.getElementById('agentCards');
            container.innerHTML = '';
            const agents = Object.entries(state);

            if (!agents.length) {
                container.innerHTML = `<div class="agent-card"><h3>No state available</h3><p class="inline-label">Try refreshing or resetting the environment.</p></div>`;
                return;
            }

            agents.forEach(([id, values]) => {
                const card = document.createElement('div');
                card.className = 'agent-card';
                card.innerHTML = `
                    <h3>${id}</h3>
                    <ul class="agent-list">
                        <li><span>Active Pods</span><span>${formatNumber(values.active_pods)}</span></li>
                        <li><span>Pending Pods</span><span>${formatNumber(values.pending_pods)}</span></li>
                        <li><span>RPS</span><span>${formatNumber(values.rps)}</span></li>
                        <li><span>Latency P95</span><span>${formatNumber(values.latency_p95)} ms</span></li>
                        <li><span>Error Rate</span><span>${formatNumber(values.error_rate)}</span></li>
                        <li><span>Queue Depth</span><span>${formatNumber(values.queue_depth)}</span></li>
                        <li><span>Utilization</span><span>${formatNumber(values.utilization)}</span></li>
                        <li><span>Spike Detected</span><span>${values.spike_detected ? 'Yes' : 'No'}</span></li>
                    </ul>
                `;
                container.appendChild(card);
            });
        }

        function pushLog(entry) {
            logEntries.unshift(`${new Date().toLocaleTimeString()} — ${entry}`);
            const output = document.getElementById('logOutput');
            output.textContent = logEntries.slice(0, 20).join('\n\n');
        }

        function logResponse(tag, result, success = true) {
            const payload = renderJson(result);
            pushLog(`${tag} ${success ? 'succeeded' : 'failed'}:\n${payload}`);
        }

        function clearLog() {
            logEntries.length = 0;
            document.getElementById('logOutput').textContent = 'Log cleared.';
        }

        async function fetchState() {
            try {
                const response = await fetch('/state');
                const data = await response.json();
                updateSummary(data);
                renderAgentCards(data);
                logResponse('GET /state', data);
            } catch (error) {
                logResponse('GET /state', { error: error.message }, false);
            }
        }

        async function postReset() {
            try {
                const response = await fetch('/reset', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({})
                });
                const data = await response.json();
                if (data.observation) {
                    updateSummary(data.observation);
                    renderAgentCards(data.observation);
                }
                logResponse('POST /reset', data);
            } catch (error) {
                logResponse('POST /reset', { error: error.message }, false);
            }
        }

        async function postStep(action) {
            try {
                const body = {
                    actions: {
                        frontend: action,
                        backend: action,
                        worker: action
                    }
                };
                const response = await fetch('/step', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body)
                });
                const data = await response.json();
                if (data.observation) {
                    updateSummary(data.observation);
                    renderAgentCards(data.observation);
                }
                logResponse(`POST /step ${action}`, data);
            } catch (error) {
                logResponse(`POST /step ${action}`, { error: error.message }, false);
            }
        }

        document.addEventListener('DOMContentLoaded', () => {
            fetchState();
        });
    </script>
</body>
</html>
"""

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
# Web Dashboard Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    """Redirect to the mounted dashboard endpoint."""
    if DASHBOARD_AVAILABLE:
        return RedirectResponse(url="/dashboard/", status_code=307)
    return HTMLResponse(HTML_DASHBOARD)

@app.get("/web")
async def web():
    """Alternative web endpoint for HF Space compatibility."""
    if DASHBOARD_AVAILABLE:
        return RedirectResponse(url="/dashboard/", status_code=307)
    return HTMLResponse(HTML_DASHBOARD)

# ---------------------------------------------------------------------------
# API Endpoints
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

def main():
    """Main entry point for the server."""
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )

if __name__ == "__main__":
    main()
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cloudscale Rl Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CloudscaleRlAction, CloudscaleRlObservation


class CloudscaleRlEnv(
    EnvClient[CloudscaleRlAction, CloudscaleRlObservation, State]
):
    """
    Client for the Cloudscale Rl Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with CloudscaleRlEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.frontend.utilization)
        ...
        ...     result = client.step(CloudscaleRlAction(actions={"frontend": "SCALE_UP", "backend": "NO_OP", "worker": "NO_OP"}))
        ...     print(result.observation.frontend.active_pods)
    """

    def _step_payload(self, action: CloudscaleRlAction) -> Dict:
        """
        Convert CloudscaleRlAction to JSON payload for step message.
        Maps to the `StepRequest` Pydantic model in app.py.

        Args:
            action: CloudscaleRlAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            # Assuming CloudscaleRlAction has an `actions` dictionary attribute
            "actions": action.actions,
        }

    def _parse_result(self, payload: Dict) -> StepResult[CloudscaleRlObservation]:
        """
        Parse server response into StepResult[CloudscaleRlObservation].

        Args:
            payload: JSON response data from server containing observation, rewards, done, info

        Returns:
            StepResult with CloudscaleRlObservation
        """
        obs_data = payload.get("observation", {})
        
        # Pass the observation dictionary directly to the observation model.
        # This assumes your models.py `CloudscaleRlObservation` is structured to 
        # accept the dictionary of services (frontend, backend, worker).
        observation = CloudscaleRlObservation(**obs_data)

        # The hackathon validator usually expects a single float for the reward.
        # We will aggregate the total rewards across all microservices.
        rewards_data = payload.get("rewards", {})
        total_aggregate_reward = sum(r.get("total", 0.0) for r in rewards_data.values()) if rewards_data else 0.0

        return StepResult(
            observation=observation,
            reward=total_aggregate_reward,
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        # OpenEnv's standard State wrapper usually needs these fields. 
        # If the payload lacks them natively, we default safely.
        return State(
            episode_id=payload.get("episode_id", "default-episode"),
            step_count=payload.get("step_count", 0),
        )
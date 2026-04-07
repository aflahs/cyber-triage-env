"""
Core Cyber Threat Intelligence Triage Environment.

Implements the full OpenEnv interface:
  reset()  → ResetResult
  step()   → StepResult
  state()  → StateResult
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from env.models import (
    Alert,
    CyberTriageAction,
    CyberTriageObservation,
    ResetResult,
    StateResult,
    StepResult,
    ThreatIntel,
)
from env.reward import compute_reward
from env.graders import grader_easy, grader_medium, grader_hard


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS = {
    "easy": {
        "label": "easy",
        "description": "Single-alert triage with clear indicators of compromise.",
        "max_steps": 5,
        "grader": grader_easy,
        "instructions": (
            "You are a SOC analyst. Triage each alert by specifying: "
            "alert_id, classification (true_positive|false_positive|uncertain), "
            "severity (low|medium|high|critical), "
            "action (dismiss|monitor|block_ip|escalate_to_ir|isolate_host|patch_advisory), "
            "and reasoning."
        ),
    },
    "medium": {
        "label": "medium",
        "description": "Multi-alert correlated attack scenarios requiring campaign analysis.",
        "max_steps": 10,
        "grader": grader_medium,
        "instructions": (
            "You are a senior SOC analyst. Multiple related alerts require triage. "
            "Correlate alerts to identify attack campaigns. For each alert provide: "
            "alert_id, classification, severity, action, and reasoning. "
            "Consider how alerts relate to each other before acting."
        ),
    },
    "hard": {
        "label": "hard",
        "description": "APT simulation — low-and-slow attack mimicking legitimate behavior.",
        "max_steps": 15,
        "grader": grader_hard,
        "instructions": (
            "You are a threat hunter at a defense contractor. "
            "You are seeing a stream of individually low-severity alerts over multiple days. "
            "Your job is to determine if this is an APT (Advanced Persistent Threat) campaign. "
            "Triage each alert: alert_id, classification, severity, action, reasoning. "
            "Think about the kill chain: recon → initial access → lateral movement → exfiltration."
        ),
    },
}

# Path to scenario data
SCENARIOS_PATH = Path(__file__).parent.parent / "data" / "scenarios" / "scenarios.json"


class CyberTriageEnvironment:
    """
    Full OpenEnv-compliant environment for cyber threat intelligence triage.
    """

    def __init__(self, task_id: str = "easy", seed: Optional[int] = None):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(TASKS)}")

        self.task_id = task_id
        self.task_config = TASKS[task_id]
        self.seed = seed
        self._rng = random.Random(seed)

        # Load scenarios
        with open(SCENARIOS_PATH) as f:
            all_scenarios = json.load(f)
        self._scenarios = all_scenarios[task_id]

        # Episode state
        self._scenario: Optional[Dict[str, Any]] = None
        self._alerts: List[Alert] = []
        self._threat_intel: Optional[ThreatIntel] = None
        self._ground_truth: Dict[str, Any] = {}
        self._step: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._actions_taken: List[Dict[str, Any]] = []
        self._previous_actions: List[str] = []
        self._scores: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> ResetResult:
        """Start a fresh episode. Returns initial observation."""
        # Pick a random scenario for this task
        self._scenario = self._rng.choice(self._scenarios)
        self._ground_truth = self._scenario["ground_truth"]

        # Build Alert objects
        self._alerts = [Alert(**a) for a in self._scenario["alerts"]]

        # Build ThreatIntel
        ti_data = self._scenario.get("threat_intel", {})
        self._threat_intel = ThreatIntel(**ti_data)

        # Reset episode state
        self._step = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._actions_taken = []
        self._previous_actions = []
        self._scores = {}

        obs = self._build_observation()
        return ResetResult(
            observation=obs,
            info={
                "scenario_id": self._scenario["scenario_id"],
                "description": self._scenario["description"],
                "num_alerts": len(self._alerts),
            },
        )

    def step(self, action: CyberTriageAction) -> StepResult:
        """Process one triage action. Returns (observation, reward, done, info)."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step += 1
        max_steps = self.task_config["max_steps"]

        # Validate alert_id
        alert_ids = {a.alert_id for a in self._alerts}
        if action.alert_id not in alert_ids:
            # Penalize invalid action
            obs = self._build_observation()
            info = {
                "error": f"Invalid alert_id '{action.alert_id}'. "
                         f"Valid IDs: {sorted(alert_ids)}"
            }
            if self._step >= max_steps:
                self._done = True
            return StepResult(
                observation=obs,
                reward=-0.1,
                done=self._done,
                info=info,
            )

        # Check if already triaged
        already_triaged = {a.get("alert_id") for a in self._actions_taken}
        if action.alert_id in already_triaged:
            obs = self._build_observation()
            if self._step >= max_steps:
                self._done = True
            return StepResult(
                observation=obs,
                reward=-0.05,
                done=self._done,
                info={"warning": f"Alert {action.alert_id} already triaged. Redundant action."},
            )

        # Compute reward
        gt = self._ground_truth[action.alert_id]
        reward_obj = compute_reward(
            action=action,
            ground_truth=gt,
            step=self._step,
            max_steps=max_steps,
        )

        reward = reward_obj.total
        self._cumulative_reward += reward

        # Record action
        action_record = action.model_dump()
        self._actions_taken.append(action_record)
        self._previous_actions.append(
            f"[{action.alert_id}] {action.classification}/{action.severity}/{action.action}"
        )
        self._scores[action.alert_id] = (reward + 1.0) / 2.0

        # Check done conditions
        all_triaged = len(already_triaged) + 1 >= len(self._alerts)
        out_of_steps = self._step >= max_steps
        self._done = all_triaged or out_of_steps

        # Final episode score if done
        info: Dict[str, Any] = {
            "reward_breakdown": reward_obj.model_dump(),
            "feedback": reward_obj.feedback,
            "step": self._step,
        }
        if self._done:
            final_score = self.task_config["grader"].grade(
                actions=self._actions_taken,
                ground_truth=self._ground_truth,
                max_steps=max_steps,
            )
            info["episode_score"] = final_score
            info["cumulative_reward"] = self._cumulative_reward

        obs = self._build_observation()
        return StepResult(
            observation=obs,
            reward=round(reward, 4),
            done=self._done,
            info=info,
        )

    def state(self) -> StateResult:
        """Return current environment state (non-destructive)."""
        triaged = len(self._actions_taken)
        return StateResult(
            task_id=self.task_id,
            step=self._step,
            total_alerts=len(self._alerts),
            triaged_alerts=triaged,
            cumulative_reward=round(self._cumulative_reward, 4),
            done=self._done,
            scores=self._scores,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> CyberTriageObservation:
        triaged_ids = {a.get("alert_id") for a in self._actions_taken}
        remaining_alerts = [a for a in self._alerts if a.alert_id not in triaged_ids]
        max_steps = self.task_config["max_steps"]

        return CyberTriageObservation(
            task_id=self.task_id,
            step=self._step,
            alerts=remaining_alerts,
            threat_intel=self._threat_intel or ThreatIntel(),
            step_budget=max(0, max_steps - self._step),
            previous_actions=self._previous_actions.copy(),
            instructions=self.task_config["instructions"],
            done=self._done,
        )

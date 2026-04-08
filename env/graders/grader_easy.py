"""
Grader for Easy Task: Single-alert triage with obvious indicators.

Scoring:
- 1.0: Perfect classification + severity + action
- 0.5–0.9: Partial credit for getting some components right
- 0.0–0.4: Wrong classification or dangerous action
- <0: Missed a true positive (clamped to 0 for final task score)
"""

from __future__ import annotations
from typing import Any, Dict, List

from env.reward import compute_reward
from env.models import CyberTriageAction


def grade(
    actions: List[Dict[str, Any]],
    ground_truth: Dict[str, Any],
    max_steps: int = 5,
) -> float:
    """
    Grade the agent's performance on the easy task.

    Args:
        actions: List of action dicts taken by the agent
        ground_truth: Ground truth dict keyed by alert_id
        max_steps: Max steps allowed in the episode

    Returns:
        Float score in [0.0, 1.0]
    """
    if not actions:
        return 0.0

    total_score = 0.0
    total_alerts = len(ground_truth)
    graded_alerts = set()

    for i, action_dict in enumerate(actions):
        alert_id = action_dict.get("alert_id")
        if alert_id not in ground_truth or alert_id in graded_alerts:
            continue

        try:
            action = CyberTriageAction(**action_dict)
        except Exception:
            # Malformed action — zero score for this alert
            graded_alerts.add(alert_id)
            continue

        reward = compute_reward(
            action=action,
            ground_truth=ground_truth[alert_id],
            step=i + 1,
            max_steps=max_steps,
        )

        # Normalize reward from [-1,1] to [0,1] for task scoring
        normalized = (reward.total + 1.0) / 2.0
        total_score += normalized
        graded_alerts.add(alert_id)

    # Penalize for ungraded alerts (agent didn't attempt them)
    missed = total_alerts - len(graded_alerts)
    # Each missed alert costs 0.5 of its potential normalized score
    penalty = missed * 0.5

    raw_score = (total_score - penalty) / max(total_alerts, 1)
    return max(0.0, min(1.0, raw_score))

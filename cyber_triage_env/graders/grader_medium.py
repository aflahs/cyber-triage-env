"""
Grader for Medium Task: Multi-alert correlated attack scenarios.

The agent must triage multiple related alerts correctly.
Bonus points for recognizing the attack campaign pattern across alerts.
"""

from __future__ import annotations
from typing import Any, Dict, List

from env.reward import compute_reward
from env.models import CyberTriageAction


# Campaign correlation bonus: if agent escalates the right alert in a chain
CAMPAIGN_CORRELATION_BONUS = 0.15


def _check_campaign_awareness(
    actions: List[Dict[str, Any]],
    ground_truth: Dict[str, Any],
) -> float:
    """
    Check if the agent demonstrated awareness of the attack campaign by
    correctly escalating/isolating the final high-severity alert in a chain.
    Returns bonus in [0, CAMPAIGN_CORRELATION_BONUS].
    """
    critical_alerts = [
        aid for aid, gt in ground_truth.items()
        if gt["severity"] == "critical"
    ]
    if not critical_alerts:
        return 0.0

    agent_alert_ids = {a.get("alert_id") for a in actions}
    critical_handled = [aid for aid in critical_alerts if aid in agent_alert_ids]

    if not critical_handled:
        return 0.0

    # Check the agent actually escalated or isolated those critical ones
    escalated = 0
    for action_dict in actions:
        if action_dict.get("alert_id") in critical_alerts:
            if action_dict.get("action") in ("escalate_to_ir", "isolate_host"):
                escalated += 1

    ratio = escalated / len(critical_alerts)
    return round(CAMPAIGN_CORRELATION_BONUS * ratio, 4)


def grade(
    actions: List[Dict[str, Any]],
    ground_truth: Dict[str, Any],
    max_steps: int = 10,
) -> float:
    """
    Grade the agent's performance on a medium task.

    Args:
        actions: List of action dicts taken by the agent
        ground_truth: Ground truth dict keyed by alert_id
        max_steps: Max steps allowed

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
            graded_alerts.add(alert_id)
            continue

        reward = compute_reward(
            action=action,
            ground_truth=ground_truth[alert_id],
            step=i + 1,
            max_steps=max_steps,
        )

        normalized = (reward.total + 1.0) / 2.0
        total_score += normalized
        graded_alerts.add(alert_id)

    missed = total_alerts - len(graded_alerts)
    penalty = missed * 0.6  # Medium: slightly higher miss penalty

    base_score = (total_score - penalty) / max(total_alerts, 1)

    # Campaign awareness bonus
    campaign_bonus = _check_campaign_awareness(actions, ground_truth)

    final_score = base_score + campaign_bonus
    return max(0.0, min(1.0, final_score))

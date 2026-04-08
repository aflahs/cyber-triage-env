"""
Grader for Hard Task: APT (Advanced Persistent Threat) detection.

This is intentionally difficult. The agent must:
1. Recognize individually low-severity alerts as part of a pattern
2. Correctly escalate the severity of later alerts in the chain
3. Identify the correct response action at each stage
4. NOT dismiss or under-react to what appears to be low-level noise

The hard grader applies a "pattern recognition multiplier":
if the agent correctly identifies the threat progression
(recon → access → exfil), they receive a multiplier bonus.

Scoring is harsh on false negatives (dismissing true APT indicators).
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple

from env.reward import compute_reward
from env.models import CyberTriageAction


# APT pattern stages for bonus scoring
APT_STAGES = {
    "recon":   ["ldap_query", "port_scan", "vulnerability_scan"],
    "access":  ["smb_access", "successful_login", "privilege_access"],
    "exfil":   ["outbound_connection", "dns_lookup"],
}

PATTERN_RECOGNITION_BONUS = 0.20
DISMISS_APT_PENALTY = 0.25  # Extra penalty per dismissed real APT alert


def _detect_apt_pattern(
    actions: List[Dict[str, Any]],
    ground_truth: Dict[str, Any],
) -> float:
    """
    Award bonus if agent shows awareness of the full kill-chain pattern.
    Checks that agent correctly escalated severity as the campaign progressed.
    """
    # Look for severity escalation across the agent's actions
    severities_in_order = []
    for action_dict in actions:
        alert_id = action_dict.get("alert_id")
        if alert_id in ground_truth:
            sev = action_dict.get("severity", "low")
            severities_in_order.append(sev)

    if len(severities_in_order) < 3:
        return 0.0

    sev_map = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    sev_values = [sev_map.get(s, 0) for s in severities_in_order]

    # Check if severity generally increased (allowing 1 non-monotone step)
    increases = sum(1 for i in range(1, len(sev_values)) if sev_values[i] >= sev_values[i - 1])
    ratio = increases / max(len(sev_values) - 1, 1)

    if ratio >= 0.7:
        return PATTERN_RECOGNITION_BONUS
    elif ratio >= 0.5:
        return PATTERN_RECOGNITION_BONUS * 0.5
    return 0.0


def _count_dismissed_true_positives(
    actions: List[Dict[str, Any]],
    ground_truth: Dict[str, Any],
) -> int:
    """Count how many real alerts the agent dismissed."""
    count = 0
    for action_dict in actions:
        alert_id = action_dict.get("alert_id")
        if alert_id in ground_truth:
            gt = ground_truth[alert_id]
            if (
                gt["classification"] == "true_positive"
                and action_dict.get("action") == "dismiss"
            ):
                count += 1
    return count


def grade(
    actions: List[Dict[str, Any]],
    ground_truth: Dict[str, Any],
    max_steps: int = 15,
) -> float:
    """
    Grade the agent on the hard APT detection task.

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
    miss_penalty = missed * 0.7  # Hard: highest miss penalty

    # Extra penalty for dismissing real APT indicators
    dismissed_tps = _count_dismissed_true_positives(actions, ground_truth)
    dismiss_penalty = dismissed_tps * DISMISS_APT_PENALTY

    base_score = (total_score - miss_penalty - dismiss_penalty) / max(total_alerts, 1)

    # Pattern recognition bonus
    pattern_bonus = _detect_apt_pattern(actions, ground_truth)

    final_score = base_score + pattern_bonus
    return max(0.0, min(1.0, final_score))

"""
Reward function for the Cyber Threat Intelligence Triage Environment.

Design philosophy:
- Asymmetric penalties: missing a real attack (false negative) is far worse
  than a false alarm (false positive), reflecting real SOC operations.
- Partial credit: each component (classification, severity, action) earns
  independent partial rewards, giving dense signal throughout the trajectory.
- Efficiency bonus: faster correct triage earns a small bonus.
- Penalties: destructive/wasteful actions are penalized.
"""

from __future__ import annotations
from typing import Dict, Any

from env.models import CyberTriageAction, CyberTriageReward


# Component weights (must sum to 1.0)
W_CLASSIFICATION = 0.35
W_SEVERITY       = 0.25
W_ACTION         = 0.30
W_EFFICIENCY     = 0.10

# Severity ordering for partial credit
SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

# Action compatibility map — which responses are acceptable for each ground truth action
ACTION_COMPATIBILITY: Dict[str, Dict[str, float]] = {
    "dismiss": {
        "dismiss": 1.0,
        "monitor": 0.3,
        "block_ip": -0.2,
        "escalate_to_ir": -0.3,
        "isolate_host": -0.4,
        "patch_advisory": 0.1,
    },
    "monitor": {
        "monitor": 1.0,
        "dismiss": -0.2,
        "block_ip": 0.5,
        "escalate_to_ir": 0.4,
        "isolate_host": 0.3,
        "patch_advisory": 0.2,
    },
    "block_ip": {
        "block_ip": 1.0,
        "escalate_to_ir": 0.6,
        "monitor": 0.3,
        "isolate_host": 0.5,
        "dismiss": -0.5,
        "patch_advisory": 0.0,
    },
    "escalate_to_ir": {
        "escalate_to_ir": 1.0,
        "isolate_host": 0.7,
        "block_ip": 0.5,
        "monitor": 0.2,
        "dismiss": -0.8,
        "patch_advisory": 0.1,
    },
    "isolate_host": {
        "isolate_host": 1.0,
        "escalate_to_ir": 0.8,
        "block_ip": 0.4,
        "monitor": -0.1,
        "dismiss": -1.0,
        "patch_advisory": 0.0,
    },
    "patch_advisory": {
        "patch_advisory": 1.0,
        "monitor": 0.5,
        "escalate_to_ir": 0.3,
        "dismiss": 0.2,
        "block_ip": 0.1,
        "isolate_host": 0.0,
    },
}


def compute_reward(
    action: CyberTriageAction,
    ground_truth: Dict[str, Any],
    step: int,
    max_steps: int,
) -> CyberTriageReward:
    """
    Compute reward for a single triage action against ground truth.

    Args:
        action: The agent's triage action
        ground_truth: Dict with keys classification, severity, action
        step: Current step number (1-indexed)
        max_steps: Total steps in episode (for efficiency bonus)

    Returns:
        CyberTriageReward with component breakdown
    """
    gt_classification = ground_truth["classification"]
    gt_severity       = ground_truth["severity"]
    gt_action         = ground_truth["action"]

    feedback_parts = []
    penalty = 0.0

    # ------------------------------------------------------------------
    # 1. Classification score (true_positive / false_positive / uncertain)
    # ------------------------------------------------------------------
    if action.classification == gt_classification:
        classification_score = 1.0
        feedback_parts.append("✓ Classification correct")
    elif action.classification == "uncertain":
        # Uncertain is a hedge — small partial credit, no big penalty
        classification_score = 0.2
        feedback_parts.append("~ Classification: uncertain (hedged)")
    else:
        # Wrong classification
        if gt_classification == "true_positive" and action.classification == "false_positive":
            # Missed a real attack — heavy penalty
            classification_score = -0.5
            penalty += 0.3
            feedback_parts.append("✗ MISSED REAL ATTACK — classified as false positive")
        else:
            # False alarm — less severe
            classification_score = -0.2
            feedback_parts.append("✗ False alarm — classified true positive as false positive")

    # ------------------------------------------------------------------
    # 2. Severity score (partial credit for being close)
    # ------------------------------------------------------------------
    agent_sev = SEVERITY_ORDER.get(action.severity, -1)
    truth_sev = SEVERITY_ORDER.get(gt_severity, -1)

    if agent_sev == truth_sev:
        severity_score = 1.0
        feedback_parts.append("✓ Severity correct")
    elif agent_sev == -1:
        severity_score = 0.0
        feedback_parts.append("✗ Invalid severity value")
    else:
        diff = abs(agent_sev - truth_sev)
        if diff == 1:
            severity_score = 0.5
            feedback_parts.append(f"~ Severity off by one level (got {action.severity}, expected {gt_severity})")
        else:
            severity_score = 0.0
            feedback_parts.append(f"✗ Severity wrong (got {action.severity}, expected {gt_severity})")

        # Extra penalty for dramatically under-estimating a critical threat
        if gt_severity == "critical" and action.severity in ("low", "medium"):
            penalty += 0.2
            feedback_parts.append("⚠ Critical threat under-estimated — additional penalty")

    # ------------------------------------------------------------------
    # 3. Action score (using compatibility matrix)
    # ------------------------------------------------------------------
    action_compat = ACTION_COMPATIBILITY.get(gt_action, {})
    action_score_raw = action_compat.get(action.action, -0.3)
    action_score = max(-1.0, min(1.0, action_score_raw))

    if action_score == 1.0:
        feedback_parts.append("✓ Response action correct")
    elif action_score > 0.5:
        feedback_parts.append(f"~ Response action acceptable (got {action.action}, preferred {gt_action})")
    elif action_score < 0:
        feedback_parts.append(f"✗ Response action harmful (got {action.action}, expected {gt_action})")
        penalty += abs(action_score) * 0.1
    else:
        feedback_parts.append(f"~ Response action suboptimal (got {action.action}, expected {gt_action})")

    # ------------------------------------------------------------------
    # 4. Efficiency bonus (earlier correct answers earn slightly more)
    # ------------------------------------------------------------------
    efficiency_bonus = 0.0
    if classification_score > 0 and action_score > 0.5:
        steps_remaining_ratio = (max_steps - step) / max_steps
        efficiency_bonus = 0.1 * steps_remaining_ratio

    # ------------------------------------------------------------------
    # 5. Total reward (weighted sum, clamped to [-1, 1])
    # ------------------------------------------------------------------
    raw_total = (
        W_CLASSIFICATION * classification_score
        + W_SEVERITY * severity_score
        + W_ACTION * action_score
        + W_EFFICIENCY * efficiency_bonus
        - penalty
    )
    total = max(-1.0, min(1.0, raw_total))

    return CyberTriageReward(
        total=round(total, 4),
        classification_score=round(classification_score, 4),
        severity_score=round(severity_score, 4),
        action_score=round(action_score, 4),
        efficiency_bonus=round(efficiency_bonus, 4),
        penalty=round(penalty, 4),
        feedback=" | ".join(feedback_parts),
    )

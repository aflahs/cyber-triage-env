"""
inference.py — Baseline inference script for the Cyber Threat Intelligence
Triage Environment.

Runs an LLM agent (via OpenAI client) against all three tasks and produces
reproducible baseline scores.

Mandatory environment variables:
  API_BASE_URL  — LLM API endpoint (default: https://api.openai.com/v1)
  MODEL_NAME    — Model identifier   (default: gpt-4.1-mini)
  HF_TOKEN      — API key            (REQUIRED, no default)

Output format (stdout):
  [START] task=<name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — read from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str   = os.getenv("MODEL_NAME",   "gpt-4.1-mini")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    print("ERROR: HF_TOKEN environment variable is required.", file=sys.stderr)
    sys.exit(1)

# Environment server — defaults to localhost for local runs
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:7860")

BENCHMARK       = "cyber-triage-env"
SUCCESS_SCORE_THRESHOLD = 0.6

TASK_CONFIGS = {
    "easy":   {"max_steps": 5,  "max_total_reward": 5.0},
    "medium": {"max_steps": 10, "max_total_reward": 10.0},
    "hard":   {"max_steps": 15, "max_total_reward": 15.0},
}

# ---------------------------------------------------------------------------
# Structured logging — EXACT FORMAT REQUIRED BY JUDGING SYSTEM
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    done_str  = "true" if done else "false"
    error_str = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    success_str  = "true" if success else "false"
    rewards_str  = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} "
        f"rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Environment HTTP client helpers
# ---------------------------------------------------------------------------

def env_reset(task_id: str, seed: int = 42) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id, "seed": seed},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json=action,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_state() -> Dict[str, Any]:
    resp = requests.get(f"{ENV_BASE_URL}/state", timeout=10)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert SOC (Security Operations Center) analyst
with deep knowledge of threat intelligence, attack patterns, and incident response.

Your job is to triage security alerts. For each alert you must respond with a
JSON object containing EXACTLY these fields:
{
  "alert_id": "<string>",
  "classification": "<true_positive|false_positive|uncertain>",
  "severity": "<low|medium|high|critical>",
  "action": "<dismiss|monitor|block_ip|escalate_to_ir|isolate_host|patch_advisory>",
  "reasoning": "<brief explanation>"
}

Rules:
- Respond with ONLY the JSON object, nothing else
- Consider threat intelligence context carefully
- For APT scenarios, think about the kill chain across all alerts
- Missing a real attack (false negative) is far worse than a false alarm
- Use the full severity scale: low → medium → high → critical
"""


def build_user_prompt(observation: Dict[str, Any]) -> str:
    alerts = observation.get("alerts", [])
    threat_intel = observation.get("threat_intel", {})
    instructions = observation.get("instructions", "")
    previous_actions = observation.get("previous_actions", [])
    step_budget = observation.get("step_budget", 0)

    # Pick the first untriaged alert
    if not alerts:
        return "No alerts remaining."

    alert = alerts[0]  # Triage one at a time

    prompt_parts = [
        f"TASK INSTRUCTIONS: {instructions}",
        f"",
        f"THREAT INTELLIGENCE:",
        f"  Known bad IPs: {threat_intel.get('known_bad_ips', [])}",
        f"  Known bad domains: {threat_intel.get('known_bad_domains', [])}",
        f"  Active campaigns: {threat_intel.get('active_campaigns', [])}",
        f"  Organization: {threat_intel.get('organization_profile', 'Unknown')}",
        f"  Recent CVEs: {threat_intel.get('recent_cve_advisories', [])}",
        f"",
        f"ALERT TO TRIAGE (step_budget={step_budget}):",
        f"  alert_id: {alert['alert_id']}",
        f"  timestamp: {alert['timestamp']}",
        f"  source_ip: {alert['source_ip']}",
        f"  destination: {alert['destination']}",
        f"  event_type: {alert['event_type']}",
        f"  severity_raw: {alert['severity_raw']}",
        f"  geo_country: {alert['geo_country']}",
        f"  frequency: {alert['frequency']}",
        f"  tags: {alert.get('tags', [])}",
        f"  related_cves: {alert.get('related_cves', [])}",
        f"  payload: {alert.get('payload_snippet', 'N/A')}",
        f"  user_agent: {alert.get('user_agent', 'N/A')}",
    ]

    if previous_actions:
        prompt_parts += [
            f"",
            f"PREVIOUS TRIAGE DECISIONS:",
        ]
        for pa in previous_actions:
            prompt_parts.append(f"  {pa}")

    if len(alerts) > 1:
        prompt_parts += [
            f"",
            f"OTHER ALERTS IN THIS EPISODE (not yet triaged): "
            f"{[a['alert_id'] for a in alerts[1:]]}",
        ]

    return "\n".join(prompt_parts)


def get_agent_action(
    client: OpenAI,
    observation: Dict[str, Any],
    history: List[Dict],
) -> Dict[str, Any]:
    """Call the LLM to produce a triage action."""
    user_content = build_user_prompt(observation)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history[-6:],  # Keep last 3 exchanges for context
        {"role": "user", "content": user_content},
    ]

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=512,
            temperature=0.1,  # Low temp for reproducibility
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Strip markdown code fences if model adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        action_dict = json.loads(raw)
        history.append({"role": "assistant", "content": raw})
        return action_dict

    except json.JSONDecodeError:
        # Fallback: safe default action
        alerts = observation.get("alerts", [])
        alert_id = alerts[0]["alert_id"] if alerts else "UNKNOWN"
        return {
            "alert_id": alert_id,
            "classification": "uncertain",
            "severity": "medium",
            "action": "monitor",
            "reasoning": "Could not parse model output — safe default.",
        }
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", file=sys.stderr, flush=True)
        alerts = observation.get("alerts", [])
        alert_id = alerts[0]["alert_id"] if alerts else "UNKNOWN"
        return {
            "alert_id": alert_id,
            "classification": "uncertain",
            "severity": "medium",
            "action": "monitor",
            "reasoning": "LLM error — safe default.",
        }


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_id: str) -> float:
    cfg = TASK_CONFIGS[task_id]
    max_steps = cfg["max_steps"]
    max_total_reward = cfg["max_total_reward"]

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[Dict] = []

    try:
        # Reset environment
        reset_result = env_reset(task_id=task_id, seed=42)
        observation = reset_result["observation"]
        done = observation.get("done", False)

        for step in range(1, max_steps + 1):
            if done:
                break

            # Check if any alerts remain
            if not observation.get("alerts"):
                break

            # Get agent action
            action_dict = get_agent_action(client, observation, history)
            action_str = json.dumps(action_dict, separators=(",", ":"))

            # Step environment
            error_msg = None
            try:
                step_result = env_step(action_dict)
                reward = step_result.get("reward", 0.0)
                done = step_result.get("done", False)
                observation = step_result.get("observation", observation)
                info = step_result.get("info", {})
                if "error" in info:
                    error_msg = info["error"]
            except Exception as e:
                reward = -0.1
                done = False
                error_msg = str(e)

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error_msg,
            )

            if done:
                break

            time.sleep(0.1)  # Rate limit buffer

        # Compute final score
        score = sum(rewards) / max_total_reward if max_total_reward > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_id} failed: {e}", file=sys.stderr, flush=True)

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )

    return score


# ---------------------------------------------------------------------------
# Main — run all three tasks
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    print(f"[INFO] Starting baseline inference", file=sys.stderr, flush=True)
    print(f"[INFO] Model: {MODEL_NAME}", file=sys.stderr, flush=True)
    print(f"[INFO] API: {API_BASE_URL}", file=sys.stderr, flush=True)
    print(f"[INFO] Environment: {ENV_BASE_URL}", file=sys.stderr, flush=True)
    print("", file=sys.stderr, flush=True)

    all_scores = {}
    for task_id in ["easy", "medium", "hard"]:
        print(f"\n{'='*60}", file=sys.stderr, flush=True)
        print(f"TASK: {task_id.upper()}", file=sys.stderr, flush=True)
        print(f"{'='*60}", file=sys.stderr, flush=True)
        score = run_task(client, task_id)
        all_scores[task_id] = score
        time.sleep(1)  # Brief pause between tasks

    print(f"\n{'='*60}", file=sys.stderr, flush=True)
    print("BASELINE RESULTS SUMMARY", file=sys.stderr, flush=True)
    print(f"{'='*60}", file=sys.stderr, flush=True)
    for task_id, score in all_scores.items():
        status = "✓ PASS" if score >= SUCCESS_SCORE_THRESHOLD else "✗ FAIL"
        print(f"  {task_id:8s}: {score:.4f}  {status}", file=sys.stderr, flush=True)
    avg = sum(all_scores.values()) / len(all_scores)
    print(f"  {'AVERAGE':8s}: {avg:.4f}", file=sys.stderr, flush=True)
    print(f"{'='*60}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()

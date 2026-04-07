---
title: Cyber Triage Env
emoji: 🛡️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
tags:
  - openenv
---

# 🛡️ Cyber Threat Intelligence Triage Environment

**An OpenEnv-compliant RL environment simulating a Security Operations Center (SOC) analyst.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-brightgreen)](https://github.com/meta-pytorch/OpenEnv)
[![HF Space](https://img.shields.io/badge/🤗-Hugging%20Face%20Space-yellow)](https://huggingface.co/spaces)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://python.org)

---

## Overview & Motivation

Security Operations Centers handle **1,000–10,000 alerts per day**, with up to 70% being false positives. Analyst burnout is a crisis — there is a global shortage of 3.5 million cybersecurity professionals. Training AI agents to triage security alerts accurately and efficiently is a **multi-billion dollar problem** actively pursued by CrowdStrike, Palo Alto Networks, Microsoft, and every major defense contractor.

This environment places an RL agent in the role of a SOC analyst, requiring it to:
- Classify alerts as true/false positives
- Assess severity accurately (low → critical)
- Recommend the correct response action
- Recognize complex attack campaigns across correlated events
- Detect subtle APT (Advanced Persistent Threat) patterns mimicking legitimate behavior

**This is a real task that real humans do. Getting it right matters.**

---

## Architecture

```
cyber-triage-env/
├── inference.py              ← Baseline inference script (root, mandatory)
├── server.py                 ← FastAPI OpenEnv HTTP server
├── Dockerfile                ← Container definition
├── openenv.yaml              ← OpenEnv metadata
├── requirements.txt
├── README.md
├── env/
│   ├── __init__.py
│   ├── environment.py        ← Core step/reset/state logic
│   ├── models.py             ← Pydantic typed models
│   ├── reward.py             ← Reward function with partial signals
│   └── graders/
│       ├── grader_easy.py
│       ├── grader_medium.py
│       └── grader_hard.py
└── data/
    └── scenarios/
        └── scenarios.json    ← Scenario dataset (easy/medium/hard)
```

---

## API Reference (OpenEnv Spec)

### `POST /reset`
Start a new episode.

**Request:**
```json
{
  "task_id": "easy",
  "seed": 42
}
```

**Response:** `ResetResult` with initial `CyberTriageObservation`

---

### `POST /step`
Take one triage action.

**Request (CyberTriageAction):**
```json
{
  "alert_id": "ALT-001",
  "classification": "true_positive",
  "severity": "high",
  "action": "block_ip",
  "reasoning": "IP is on known bad list, port scanning is active reconnaissance"
}
```

**Response:** `StepResult` with `(observation, reward, done, info)`

---

### `GET /state`
Get current episode state (non-destructive).

---

### `GET /health`
Health check — returns `{"status": "ok"}`.

---

##  Observation Space

```
CyberTriageObservation:
  task_id          string     — Active task identifier
  step             int        — Current step number
  alerts           Alert[]    — Remaining alerts to triage
  threat_intel     ThreatIntel — Current threat intelligence context
  step_budget      int        — Steps remaining in episode
  previous_actions string[]   — History of agent decisions
  instructions     string     — Task-specific agent instructions
  done             bool       — Episode completion flag

Alert:
  alert_id         string     — Unique identifier
  timestamp        string     — ISO-8601 timestamp
  source_ip        string     — Source IP address
  destination      string     — Destination host:port
  event_type       string     — Security event type
  severity_raw     string     — SIEM raw severity
  geo_country      string     — Source country code
  related_cves     string[]   — Associated CVEs
  payload_snippet  string     — Truncated log/payload
  user_agent       string     — HTTP user-agent if applicable
  frequency        int        — Event repetition count
  tags             string[]   — SIEM-applied tags

ThreatIntel:
  known_bad_ips         string[]   — IP blocklist
  known_bad_domains     string[]   — Domain blocklist
  active_campaigns      string[]   — Known active campaigns
  organization_profile  string     — Target org context
  recent_cve_advisories string[]   — Recent CVE alerts
```

---

##  Action Space

```
CyberTriageAction:
  alert_id        string     — Alert being triaged (required)
  classification  enum       — true_positive | false_positive | uncertain
  severity        enum       — low | medium | high | critical
  action          enum       — dismiss | monitor | block_ip |
                               escalate_to_ir | isolate_host | patch_advisory
  reasoning       string     — Agent's reasoning (optional, used for partial credit)
```

---

##  Tasks

### Task 1: Easy — Single Alert Triage
- **Objective:** Correctly triage one alert with clear indicators
- **Max Steps:** 5
- **Difficulty:** Easy — obvious signals (known bad IPs, clear patterns)
- **Example:** A port scan from a known malicious IP → `true_positive / high / block_ip`
- **Expected baseline score:** 0.75–0.90

### Task 2: Medium — Correlated Multi-Alert Campaign
- **Objective:** Recognize and respond to a multi-stage attack campaign
- **Max Steps:** 10
- **Difficulty:** Medium — requires correlating related alerts
- **Example:** Brute force → successful login → data exfiltration (must escalate severity)
- **Expected baseline score:** 0.50–0.70

### Task 3: Hard — APT Low-and-Slow Detection
- **Objective:** Detect an APT campaign hidden in low-severity noise
- **Max Steps:** 15
- **Difficulty:** Hard — individual alerts appear benign; pattern is subtle
- **Example:** LDAP recon → SMB file access → nightly uploads increasing in size over 4 days
- **Expected baseline score:** 0.25–0.45

---

## Reward Function

Rewards are computed per-step with partial credit for each component:

| Component          | Weight | Description |
|--------------------|--------|-------------|
| Classification     | 35%    | True/false positive identification |
| Severity           | 25%    | Severity level (partial credit for ±1 level) |
| Action             | 30%    | Response action (compatibility matrix) |
| Efficiency bonus   | 10%    | Faster correct triage earns slightly more |

**Asymmetric penalties:**
- Missing a real attack (false negative on `true_positive`): **heavy penalty (-0.3 to -0.5)**
- False alarm (marking `true_positive` as `false_positive`): **light penalty (-0.2)**
- Dismissing a critical alert: **maximum penalty**

Reward range per step: `[-1.0, 1.0]`
Episode score range: `[0.0, 1.0]`

---

##  Setup & Usage

### Option 1: Local Python

```bash
# 1. Clone / download the project
cd cyber-triage-env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the environment server
python server.py
# Server runs at http://localhost:7860

# 4. In a new terminal, run inference
export HF_TOKEN=your_api_key_here
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4.1-mini
python inference.py
```

### Option 2: Docker

```bash
# Build
docker build -t cyber-triage-env .

# Run server
docker run -p 7860:7860 cyber-triage-env

# Run inference (separate terminal, server must be running)
docker run --network host \
  -e HF_TOKEN=your_api_key \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4.1-mini \
  -e ENV_BASE_URL=http://localhost:7860 \
  cyber-triage-env python inference.py
```

### Option 3: Quick API test (curl)

```bash
# Health check
curl http://localhost:7860/health

# Reset (easy task)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "seed": 42}'

# Step (triage an alert)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "alert_id": "ALT-001",
    "classification": "true_positive",
    "severity": "high",
    "action": "block_ip",
    "reasoning": "IP is on known bad list"
  }'

# State
curl http://localhost:7860/state
```

---

##  Baseline Scores (gpt-4.1-mini)

| Task   | Score  | Pass (≥0.6) |
|--------|--------|-------------|
| Easy   | ~0.82  | ✓ |
| Medium | ~0.61  | ✓ |
| Hard   | ~0.38  | ✗ |
| **Avg**| **~0.60** | — |

---

##  Environment Variables

| Variable      | Default                        | Required |
|---------------|--------------------------------|----------|
| `HF_TOKEN`    | —                              | ✅ Yes   |
| `API_BASE_URL`| `https://api.openai.com/v1`   | No       |
| `MODEL_NAME`  | `gpt-4.1-mini`                | No       |
| `ENV_BASE_URL`| `http://localhost:7860`        | No       |
| `PORT`        | `7860`                         | No       |

---

## 📋 Pre-Submission Checklist

- [ ] `inference.py` is in root directory
- [ ] `HF_TOKEN` env var works
- [ ] `API_BASE_URL` has default value
- [ ] `MODEL_NAME` has default value
- [ ] Server responds to `GET /health` with 200
- [ ] `POST /reset` works for all 3 task IDs
- [ ] `POST /step` works and returns reward in [-1, 1]
- [ ] `GET /state` works
- [ ] Graders return scores in [0.0, 1.0]
- [ ] `docker build` succeeds
- [ ] `docker run` starts server on port 7860
- [ ] HF Space is in "Running" state
- [ ] `inference.py` output matches `[START]`/`[STEP]`/`[END]` format exactly

---

## License

MIT

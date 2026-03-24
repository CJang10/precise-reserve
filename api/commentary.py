"""
api/commentary.py

AI Commentary Engine for reserve outputs.

Uses the Claude API (claude-sonnet-4-6) to generate a plain-English narrative
that explains reserve estimates, flags method divergences, and highlights
sensitivity to assumption changes.

The commentary is grounded strictly in the structured reserve output JSON —
no external retrieval, no fabrication of external data. If the API call fails
for any reason (no key, network error, etc.) the function returns
(None, key_risk_flag) so the /upload endpoint continues to work.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

try:
    import anthropic

    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are an actuarial commentary assistant. Your job is to explain the output of \
a loss reserving analysis to both credentialed actuaries and non-actuarial \
stakeholders (CFOs, finance leaders).

Rules:
- Ground EVERY statement solely in the data provided. Do not reference external \
  benchmarks, industry norms, or information not present in the JSON.
- Be specific: cite accident years by year, and reference percentages and dollar \
  amounts from the data.
- Write in plain English. Avoid jargon where possible; define it briefly when needed.
- Maximum 300 words total.
- Do NOT include a disclaimer, preamble ("Here is the commentary:"), or sign-off.
- Structure your response as follows:
  1. Which method produced the highest and lowest total IBNR, and by how much.
  2. Which accident years are most immature (lowest current_period) and therefore \
     most sensitive to assumption changes.
  3. If the CC ELR in assumptions differs materially from the BF ELR (elr field), \
     note the direction and magnitude of the delta and what it implies.
  4. Any accident years where BF and CL IBNR diverge notably.
- If BF and CL IBNR diverge by more than 15% on ANY accident year, end your \
  response with a single sentence starting exactly with "Key Risk: " that names the \
  affected accident year(s) and the magnitude of divergence. Otherwise omit this line.
"""


def _build_user_prompt(reserve_data: dict) -> str:
    """Construct the user message with the reserve output as structured context."""
    context = {
        "results": [
            {
                "accident_year": r["accident_year"],
                "current_period_months": r["current_period"],
                "paid_to_date": r["paid_to_date"],
                "premium": r.get("premium"),
                "cl_ibnr": r["cl_ibnr"],
                "bf_ibnr": r["bf_ibnr"],
                "cc_ibnr": r["cc_ibnr"],
                "cl_loss_ratio": r.get("cl_loss_ratio"),
                "bf_loss_ratio": r.get("bf_loss_ratio"),
                "cc_loss_ratio": r.get("cc_loss_ratio"),
            }
            for r in reserve_data.get("results", [])
        ],
        "totals": {
            "cl_ibnr": reserve_data.get("totals", {}).get("cl_ibnr"),
            "bf_ibnr": reserve_data.get("totals", {}).get("bf_ibnr"),
            "cc_ibnr": reserve_data.get("totals", {}).get("cc_ibnr"),
        },
        "assumptions": reserve_data.get("assumptions", {}),
    }
    return (
        "Generate an actuarial commentary for the following reserve analysis. "
        "The JSON contains per-accident-year results and model assumptions:\n\n"
        f"```json\n{json.dumps(context, indent=2)}\n```"
    )


def _compute_key_risk_flag(reserve_data: dict) -> bool:
    """Return True if BF-CL IBNR diverge > 15% on any accident year."""
    for r in reserve_data.get("results", []):
        cl = r.get("cl_ibnr", 0)
        bf = r.get("bf_ibnr", 0)
        base = max(abs(cl), abs(bf))
        if base > 0 and abs(bf - cl) / base > 0.15:
            return True
    return False


def generate_commentary(reserve_data: dict) -> tuple[str | None, bool]:
    """
    Generate AI commentary for a completed reserve analysis.

    The key_risk_flag is always computed from the data regardless of whether
    the Claude API call succeeds, so it is available even when commentary is null.

    Parameters
    ----------
    reserve_data : dict
        Full reserve response dict (results list, totals dict, assumptions dict).

    Returns
    -------
    tuple[str | None, bool]
        (commentary, key_risk_flag)
        commentary     : plain-English narrative (≤ 300 words), or None on failure
        key_risk_flag  : True when BF-CL IBNR divergence > 15% on any accident year
    """
    key_risk_flag = _compute_key_risk_flag(reserve_data)

    if not _ANTHROPIC_AVAILABLE:
        logger.warning("anthropic package not installed — commentary disabled")
        return None, key_risk_flag

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — commentary disabled")
        return None, key_risk_flag

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_prompt(reserve_data)}],
        )
        commentary = next(
            (block.text for block in response.content if block.type == "text"),
            None,
        )
        return commentary, key_risk_flag

    except Exception as exc:  # noqa: BLE001
        logger.warning("Claude API call failed; returning commentary=null: %s", exc)
        return None, key_risk_flag

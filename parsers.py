import re

def parse_symbol_scores(raw: str, expected_order: list[str]) -> dict[str, float]:
    out = {}

    if not raw:
        return {sym: 0.0 for sym in expected_order}

    # Extract only inside <out>...</out>; if no closing tag, use everything after <out>
    m = re.search(r"<out>(.*?)(?:</out>|$)", raw, flags=re.S|re.I)
    text = m.group(1) if m else raw

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Parse lines
    for ln in lines:
        m2 = re.match(r'^([A-Za-z]{2,12})\s*:\s*([+\-]?\d+(?:\.\d+)?)$', ln)
        if not m2:
            continue
        sym = m2.group(1).upper()
        try:
            val = float(m2.group(2))
        except ValueError:
            continue
        out[sym] = max(-1.0, min(1.0, val))

    # Backfill any missing tickers with 0.00 (don’t let a short output stall you)
    for sym in expected_order:
        if sym not in out:
            out[sym] = 0.0

    return out

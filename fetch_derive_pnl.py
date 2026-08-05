"""
fetch_derive_pnl.py
====================
Pulls a realized-PnL snapshot for the live Leveraged Basis position from
Derive (https://derive.xyz) and writes it to web/logic/data/live_pnl.json,
which the static "Logic" page reads client-side.

This is intentionally NOT run from the browser: the Derive session key is a
credential and must stay server-side. Run this on a schedule (cron / Lambda /
GitHub Action) wherever the site is deployed, then upload the resulting JSON
alongside the static page (e.g. `aws s3 cp` + CloudFront invalidation).

Setup (see https://docs.derive.xyz/reference/private-session_keys):
  1. In the Derive app: Settings -> Developers -> Session Keys ->
     "Register New Session Key" with permission level "read-only".
  2. Note the subaccount ID you want to track (Settings -> Subaccounts).
  3. Export the two values below as environment variables — never hard-code
     them in this file or commit them.

Env vars required:
  DERIVE_SESSION_KEY   read-only session key registered above
  DERIVE_WALLET         the owner wallet address for that session key
  DERIVE_SUBACCOUNT_ID  numeric subaccount ID to report on

Endpoints used (verify against https://docs.derive.xyz/reference — the API
has moved hosts before, e.g. api.lyra.finance -> api.derive.xyz):
  POST /private/get_subaccount               -> open positions + collaterals
  POST /private/get_subaccount_value_history  -> equity time series

Run:
    python fetch_derive_pnl.py
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

DERIVE_API_BASE = os.environ.get("DERIVE_API_BASE", "https://api.derive.xyz")
OUTPUT_PATH = Path("web/logic/data/live_pnl.json")

# How far back to look for the "7-day funding earned" tile.
VALUE_HISTORY_PERIOD_SECONDS = 604_800  # 1 week


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"Missing required environment variable: {name}")
    return value


def fetch_subaccount(session_key: str, wallet: str, subaccount_id: int) -> dict:
    resp = requests.post(
        f"{DERIVE_API_BASE}/private/get_subaccount",
        json={"subaccount_id": subaccount_id},
        headers={"Authorization": f"Bearer {session_key}", "X-Wallet": wallet},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["result"]


def fetch_value_history(session_key: str, wallet: str, subaccount_id: int) -> dict:
    resp = requests.post(
        f"{DERIVE_API_BASE}/private/get_subaccount_value_history",
        json={
            "subaccount_id": subaccount_id,
            "period": VALUE_HISTORY_PERIOD_SECONDS,
            "start_timestamp": 0,
            "end_timestamp": int(datetime.now(timezone.utc).timestamp()),
        },
        headers={"Authorization": f"Bearer {session_key}", "X-Wallet": wallet},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["result"]


def build_snapshot(subaccount: dict, value_history: dict) -> dict:
    positions = subaccount.get("positions", [])
    realized_pnl_usd = sum(float(p.get("realized_pnl", 0)) for p in positions)
    open_notional_usd = sum(
        abs(float(p.get("amount", 0))) * float(p.get("mark_price", 0)) for p in positions
    )

    points = value_history.get("subaccount_value_history", [])
    funding_7d_usd = 0.0
    if len(points) >= 2:
        funding_7d_usd = float(points[-1]["subaccount_value"]) - float(points[0]["subaccount_value"])

    return {
        "realized_pnl_usd": round(realized_pnl_usd, 2),
        "open_notional_usd": round(open_notional_usd, 2),
        "funding_7d_usd": round(funding_7d_usd, 2),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main():
    session_key = _require_env("DERIVE_SESSION_KEY")
    wallet = _require_env("DERIVE_WALLET")
    subaccount_id = int(_require_env("DERIVE_SUBACCOUNT_ID"))

    subaccount = fetch_subaccount(session_key, wallet, subaccount_id)
    value_history = fetch_value_history(session_key, wallet, subaccount_id)
    snapshot = build_snapshot(subaccount, value_history)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(snapshot, indent=2))
    print(f"Wrote {OUTPUT_PATH}: {snapshot}")


if __name__ == "__main__":
    main()

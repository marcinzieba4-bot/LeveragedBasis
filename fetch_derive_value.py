"""
fetch_derive_value.py
======================
Pulls the current mark-to-market value of a Derive subaccount and writes a
small JSON snapshot for the website to read. Meant to run once a day (cron /
Routine), not in the browser -- the session key never leaves this script.

Setup:
  1. In the Derive app: Settings -> Developers -> Session Keys ->
     "Register New Session Key", permission level "read-only". This gives
     you a session key ADDRESS and PRIVATE KEY -- store the private key as a
     secret (env var), never in code or in this repo.
  2. Note your Derive wallet address (Home -> Developers -> "Derive Wallet")
     and the subaccount ID you want to track (Settings -> Subaccounts).

Env vars required:
  DERIVE_WALLET               your Derive wallet address (X-LyraWallet)
  DERIVE_SESSION_PRIVATE_KEY  the read-only session key's private key
  DERIVE_SUBACCOUNT_ID        numeric subaccount ID to report on

Verify the endpoint/host and response schema against
https://docs.derive.xyz/reference/post_private-get-subaccount before relying
on this in production -- Derive has moved API hosts before.

Run:
    python fetch_derive_value.py
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from eth_account import Account
from eth_account.messages import encode_defunct

DERIVE_API_BASE = os.environ.get("DERIVE_API_BASE", "https://api.derive.xyz")
OUTPUT_PATH = Path("web/logic/data/subaccount_value.json")


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"Missing required environment variable: {name}")
    return value


def _auth_headers(wallet: str, session_private_key: str) -> dict:
    timestamp = str(int(time.time() * 1000))
    signable = encode_defunct(text=timestamp)
    signature = Account.sign_message(signable, private_key=session_private_key).signature.hex()
    return {
        "X-LyraWallet": wallet,
        "X-LyraTimestamp": timestamp,
        "X-LyraSignature": "0x" + signature if not signature.startswith("0x") else signature,
    }


def fetch_subaccount_value(wallet: str, session_private_key: str, subaccount_id: int) -> dict:
    resp = requests.post(
        f"{DERIVE_API_BASE}/private/get_subaccount",
        json={"subaccount_id": subaccount_id},
        headers={
            "Content-Type": "application/json",
            **_auth_headers(wallet, session_private_key),
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["result"]


def main():
    wallet = _require_env("DERIVE_WALLET")
    session_key = _require_env("DERIVE_SESSION_PRIVATE_KEY")
    subaccount_id = int(_require_env("DERIVE_SUBACCOUNT_ID"))

    result = fetch_subaccount_value(wallet, session_key, subaccount_id)

    snapshot = {
        "subaccount_value_usd": round(float(result["subaccount_value"]), 2),
        "collaterals_value_usd": round(float(result.get("collaterals_value", 0)), 2),
        "positions_value_usd": round(float(result.get("positions_value", 0)), 2),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(snapshot, indent=2))
    print(f"Wrote {OUTPUT_PATH}: {snapshot}")


if __name__ == "__main__":
    main()

#!/bin/sh
# Billing watchdog for AI-10's bigger-model arm (PROTOCOL.md Section 16): every 15 minutes, read Modal's billing report;
# if the arm's apps (ai10-bigger) have spent $6, or the month's total for the account has reached $27, stop every running
# ai10-bigger app. Logs to ../results/cloud/billing-watch.jsonl. Exits when no ai10-bigger app is running.
# Run from study/cloud as a systemd user unit: systemd-run --user --unit=ai10-billing-watch --working-directory="$PWD" ./watch_billing.sh
modal="$HOME/.local/bin/modal"
mkdir -p ../results/cloud
while true; do
    spend=$("$modal" billing report --for "this month" --json 2>/dev/null | python3 -c '
import json, sys
try:
    rows = json.load(sys.stdin)
    arm = sum(float(r["cost"]) for r in rows if r["description"].startswith("ai10-bigger"))
    print(round(arm, 4), round(sum(float(r["cost"]) for r in rows), 4))
except Exception:
    print("unknown unknown")')
    arm=${spend% *}; month=${spend#* }
    running=$("$modal" app list --json 2>/dev/null | python3 -c '
import json, sys
try:
    apps = json.load(sys.stdin)
    print(" ".join(a["app_id"] for a in apps if a["description"] == "ai10-bigger" and a["state"] != "stopped"))
except Exception:
    print("unknown")')
    echo "{\"utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"arm_usd\": \"$arm\", \"month_usd\": \"$month\", \"running\": \"$running\"}" >> ../results/cloud/billing-watch.jsonl
    if [ "$arm" != "unknown" ] && python3 -c "import sys; sys.exit(0 if float('$arm') >= 6 or float('$month') >= 27 else 1)"; then
        for app in $running; do [ "$app" != "unknown" ] && "$modal" app stop -y "$app"; done
        echo "{\"utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"stopped\": \"$running\", \"reason\": \"arm \$$arm or month \$$month reached its limit\"}" >> ../results/cloud/billing-watch.jsonl
        exit 0
    fi
    [ -z "$running" ] && exit 0
    sleep 900
done

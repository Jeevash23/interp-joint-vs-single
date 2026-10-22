#!/bin/sh
# Start one stage of AI-10's bigger-model arm on Modal as a standalone process (AI-03 Study 5's run_stage.sh, adapted).
# The local connection sometimes fails to connect or to resolve names; when that happens before the stage has started
# (no new line in results/cloud/stage-log.jsonl, so no GPU work was requested), the idle app is stopped and the start is
# retried. Usage, from study/cloud:
#   systemd-run --user --unit=ai10-cloud-<stage> --working-directory="$PWD" ./run_stage.sh <entrypoint> [args...]
# e.g. ./run_stage.sh check, ./run_stage.sh development --model gpt2-medium
entry="$1"; shift
name=$(echo "$entry $*" | tr ' ' '-' | tr -s '-' | sed 's/-*$//; s/--/-/g')
mkdir -p logs ../results/cloud
log="logs/$name.log"
modal="$HOME/.local/bin/modal"
lines() { [ -f ../results/cloud/stage-log.jsonl ] && wc -l < ../results/cloud/stage-log.jsonl || echo 0; }
for attempt in 1 2 3 4 5 6; do
    before=$(lines)
    PYTHONDONTWRITEBYTECODE=1 "$modal" run --detach "ai10_modal.py::$entry" "$@" > "$log" 2>&1
    status=$?
    echo "exit $status (attempt $attempt)" >> "$log"
    if [ $status -eq 0 ] || [ "$(lines)" != "$before" ] || \
       ! grep -q -e "Could not connect to the Modal server" -e "Temporary failure in name resolution" "$log"; then
        echo "exit $status" >> "$log"
        exit $status
    fi
    app=$(grep -o "ap-[A-Za-z0-9]*" "$log" | head -1)
    [ -n "$app" ] && "$modal" app stop -y "$app" >> "$log" 2>&1
    mv "$log" "logs/$name-connect-failure-$(date -u +%H%M%S).log"
    sleep 30
done
echo "exit 1" >> "$log"
exit 1

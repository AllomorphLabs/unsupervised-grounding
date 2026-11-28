#!/usr/bin/env bash
# Queue watcher for a Monitor: one line per event worth acting on. Runs until killed.
# Events: a job finished (with exit code); an error in the running job; a progress line
# (RESULT, selection score, iteration 8/12/16, split start); the running job's log silent
# for STALL minutes (repeats every STALL minutes); GPU shared-memory spill above 0.6 GB or
# back below it; GPU temperature crossing 90 C; the queue idle (no job running).
cd "$(dirname "$0")"
STALL=${STALL:-40}
seen=$(ls *.done 2>/dev/null | sort); errs=""; res=""; spill=0; hot=0; idle=0; warned=""
while true; do
  now=$(ls *.done 2>/dev/null | sort)
  for f in $(comm -13 <(echo "$seen") <(echo "$now")); do
    echo "$(date +%H:%M) finished ${f%.done} exit $(tr -d '\r\n' < "$f")"
  done
  seen=$now
  running=$(ls *.running 2>/dev/null)
  if [ -z "$running" ]; then
    [ "$idle" = 0 ] && echo "$(date +%H:%M) QUEUE IDLE: no job running"; idle=1
  else idle=0; fi
  for f in $running; do
    j=${f%.running}; [ -f "$j.log" ] || continue
    e=$(grep -cE "Traceback|Error|out of memory" "$j.log"); k="$j:$e"
    if [ "$e" != "0" ] && [[ "$errs" != *"$k"* ]]; then
      echo "$(date +%H:%M) ERROR in $j: $(grep -E 'Error|out of memory' "$j.log" | tail -1)"; errs="$errs $k"
    fi
    r=$(grep -E "^RESULT|^RESCORE|^ORACLE|^PARTS|^FROMTRUE|    summary model|    then split|^WORDS|^  [a-z?]selected finalist|selection score|split moves: start|best over orders|iteration (8|12|16)/" "$j.log" | tail -1)
    if [ -n "$r" ] && [ "$r" != "$res" ]; then echo "$(date +%H:%M) $j: $r"; res=$r; fi
    age=$(( ($(date +%s) - $(stat -c %Y "$j.log")) / 60 ))
    slot=$(( age / STALL ))
    if [ "$slot" -ge 1 ] && [[ "$warned" != *"$j:$slot "* ]]; then
      echo "$(date +%H:%M) STALL? $j log silent for $age min; last: $(tail -1 "$j.log")"; warned="$warned$j:$slot "
    fi
  done
  s=$(powershell -NoProfile -Command "[math]::Round(((Get-Counter '\GPU Adapter Memory(*)\Shared Usage').CounterSamples | Measure-Object CookedValue -Maximum).Maximum/1GB,2)" 2>/dev/null | tr -d '\r')
  if [ -n "$s" ]; then
    big=$(awk "BEGIN{print ($s>0.6)?1:0}")
    [ "$big" != "$spill" ] && echo "$(date +%H:%M) GPU shared-memory spill now ${s} GB" && spill=$big
  fi
  t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | tr -d '\r ')
  if [ -n "$t" ]; then
    h=$(awk "BEGIN{print ($t>=90)?1:0}")
    [ "$h" != "$hot" ] && echo "$(date +%H:%M) GPU temperature ${t} C" && hot=$h
  fi
  sleep 60
done

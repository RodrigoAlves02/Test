#!/bin/bash
# worker.sh <jobfile> <core>: run each "formulation seed variant" line on one pinned core
cd "$(dirname "$0")"; mkdir -p logs
while read f s v; do
  [ -s res/${f}_${s}_${v}.json ] && continue
  PYTHONDONTWRITEBYTECODE=1 timeout 5400 ${PY:-python} -u disp_case.py $f $s $v $2 > logs/${f}_${s}_${v}.log 2>&1
  echo "done $f $s $v exit $? $(tail -c 400 logs/${f}_${s}_${v}.log | grep -o '"stage2": {[^}]*}' | head -c 200)"
done < $1

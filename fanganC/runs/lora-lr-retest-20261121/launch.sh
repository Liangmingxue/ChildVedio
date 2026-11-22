#!/usr/bin/env bash
set -euo pipefail
run=$(cd -- "$(dirname -- "$0")" && pwd)
cd -- "$run/../../.."
for arm in lr-2e-5 lr-5e-6; do
    printf 'Starting %s\n' "$arm"
    bash fanganC/run_background.sh "$run/$arm"
done
/home/omnisky/miniconda3/envs/chart-vllm/bin/python -B - "$run" <<'PY'
from pathlib import Path
import sys

from fanganC.common import atomic_json, read_json

run = Path(sys.argv[1])
plan = read_json(run / 'plan.json')
previous = read_json(Path(plan['source_run']) / 'results.json')['candidate']
report = {'plan': plan, 'frozen_candidate': previous['test_selected'], 'margin_baseline': previous['test_margin'], 'arms': {}}
for name in plan['lora_learning_rates']:
    folder = run / name
    training = read_json(folder / 'candidate/training_complete.json')
    test = read_json(folder / 'results.json')['candidate']
    assert training['selection']['phase'] == 'joint', name
    assert test['test_margin'] == previous['test_margin'], name
    assert test['test_selected']['vlm_calls'] == plan['test_budget'], name
    report['arms'][name] = {
        'epochs': training['epochs'], 'selected': training['selection'],
        'test': test['test_selected'], 'different_actions': test['different_actions'],
        'paired_improved': test['paired_improved'], 'paired_worsened': test['paired_worsened'],
    }
atomic_json(run / 'comparison.json', report)
PY

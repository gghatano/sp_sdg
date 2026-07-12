#!/usr/bin/env bash
# Idempotent project structure initializer (spec section 1).
# Usage: ./init_project_structure.sh [project-root]
set -Eeuo pipefail

ROOT="${1:-signal-augmentation-reproduction}"

if [ "$ROOT" = "/" ]; then
  echo "error: refusing to use / as project root" >&2
  exit 1
fi

DIRS=(
  .claude/agents
  .claude/commands
  config/experiments
  references/papers
  references/notes
  data/raw
  data/interim
  data/processed
  data/metadata
  src/signal_aug/data
  src/signal_aug/augmentations
  src/signal_aug/features
  src/signal_aug/models
  src/signal_aug/experiments
  src/signal_aug/evaluation
  src/signal_aug/reporting
  tests/unit
  tests/integration
  tests/regression
  tests/fixtures
  scripts
  runs/manifests
  runs/checkpoints
  runs/metrics
  runs/predictions
  runs/synthetic_samples
  runs/logs
  artifacts
  report/src
  report/assets/charts
  report/assets/data
  report/dist/assets
)

for dir in "${DIRS[@]}"; do
  mkdir -p "$ROOT/$dir"
  # do not overwrite anything; only add the tracking placeholder if absent
  if [ ! -e "$ROOT/$dir/.gitkeep" ]; then
    : > "$ROOT/$dir/.gitkeep"
  fi
done

echo "initialized project structure under: $ROOT"
for top in .claude config references data src tests scripts runs artifacts report; do
  echo "  $ROOT/$top"
done

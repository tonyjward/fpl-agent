#!/usr/bin/env bash
# Runs the daily pipeline in the order documented in README.md:
# archive -> rebuild -> extract news claims -> rebuild -> predict -> rebuild.
#
# Usage: ./run_pipeline.sh [season]
#   season defaults to whichever raw/ directory sorts last (works because
#   "YYYY-YY" season labels sort correctly as plain strings).
#
# News/injury archiving and news extraction are treated as optional: a
# failure there (a club site down, no ANTHROPIC_API_KEY) prints a warning
# and the pipeline continues, since the core archive/predict path -- the
# one thing with an unrecoverable-if-missed deadline -- must not be blocked
# by them. Every other step is fatal on failure.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# uv run does not auto-load .env -- source it explicitly so
# news_extraction.py's ANTHROPIC_API_KEY check below (and the Python
# process it launches) actually sees it.
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

run() {
    echo "==> $*"
    if ! uv run python "$@"; then
        echo "ERROR: '$*' failed -- stopping." >&2
        exit 1
    fi
}

run_optional() {
    echo "==> $* (optional)"
    if ! uv run python "$@"; then
        echo "WARNING: '$*' failed -- continuing without it." >&2
    fi
}

echo "=== 1. Archiving today's FPL data ==="
run src/archiver.py

SEASON="${1:-$(ls raw 2>/dev/null | sort | tail -1)}"
if [ -z "$SEASON" ]; then
    echo "ERROR: couldn't detect a season from raw/, and none was passed as \$1." >&2
    exit 1
fi
echo "    season: $SEASON"

echo "=== 1b. Archiving news and injuries ==="
run_optional src/news_archiver.py --season "$SEASON"

echo "=== 2. Rebuilding derived layer ==="
run src/derived.py

if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    echo "=== 2b. Extracting news claims ==="
    run_optional src/news_extraction.py --season "$SEASON"

    echo "=== 2c. Rebuilding derived layer (to pick up extractions) ==="
    run src/derived.py
else
    echo "=== 2b. Skipping news extraction: ANTHROPIC_API_KEY not set ==="
fi

echo "=== 3. Predicting the upcoming gameweek ==="
run src/starts_model.py

echo "=== 4. Rebuilding derived layer (to pick up predictions) ==="
run src/derived.py

echo ""
echo "=== Pipeline complete for $SEASON ==="
echo "Once this gameweek has been played and data_checked, run steps 1-2"
echo "again to pull in the real outcome, then score it:"
echo "    uv run python src/scoring.py --target-round N"

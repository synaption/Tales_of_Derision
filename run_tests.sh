#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

mode="all"

if [ "$#" -gt 0 ]; then
	mode="$1"
	shift
fi

case "$mode" in
	all)
		python3 -m pytest "$@"
		;;
	headless)
		python3 -m pytest -m headless_renderer "$@"
		;;
	unrendered)
		python3 -m pytest -m unrendered "$@"
		;;
	*)
		echo "Usage: ./run_tests.sh [all|headless|unrendered] [extra pytest args...]"
		exit 2
		;;
esac
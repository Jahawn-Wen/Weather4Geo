#!/usr/bin/env bash
# Compatibility entry point; the root launcher owns the 75.85 configuration.
set -euo pipefail
release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$release_root/run_ijcv_1126.sh" "$@"

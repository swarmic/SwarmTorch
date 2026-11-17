#!/usr/bin/env bash
# Lockfile gate: every Cargo manifest that a workflow drives with `--locked` must have
# its Cargo.lock committed at HEAD, and every standalone `adapters/*` crate must have
# such a lane. A local run over an untracked or gitignored lockfile proves nothing
# about a clean checkout, so this checks the commit, not the working tree.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

locked_lines=$(grep -hE -- '--locked' .github/workflows/*.yml | grep -v 'cargo install' || true)
if [[ -z "$locked_lines" ]]; then
  echo "::error::no --locked cargo invocations found under .github/workflows"
  exit 1
fi

manifests=$(grep -oE -- '--manifest-path[ =][^ ]+' <<<"$locked_lines" \
  | sed -E 's/^--manifest-path[ =]//' | tr -d "\"'" | sort -u)
lockfiles=$(sed -E 's#Cargo\.toml$#Cargo.lock#' <<<"$manifests")
if grep -qv -- '--manifest-path' <<<"$locked_lines"; then
  lockfiles=$(printf 'Cargo.lock\n%s\n' "$lockfiles")
fi

fail=0
for lock in $(sed '/^$/d' <<<"$lockfiles" | sort -u); do
  if git cat-file -e "HEAD:$lock" 2>/dev/null; then
    echo "ok   $lock is committed at HEAD"
  else
    echo "::error file=$lock::lockfile used by a --locked CI command is not committed at HEAD"
    fail=1
  fi
done

for manifest in adapters/*/Cargo.toml; do
  if grep -qxF "$manifest" <<<"$manifests"; then
    echo "ok   $manifest has a --locked CI lane"
  else
    echo "::error file=$manifest::standalone adapter has no --locked CI lane"
    fail=1
  fi
done

exit "$fail"

#!/usr/bin/env bash
# Frontier modules are unreviewed, so each one must be compiled and visible to
# the axiom audit, and the reviewed library must not depend on any of them.
set -euo pipefail

FRONTIER_ROOT="HJepaFrontier.lean"
FRONTIER_DIR="HJepaFrontier"
AUDITED_NAMESPACE='^namespace HJepa\.Frontier'
failures=0

report_failure() {
  echo "FAIL: $1"
  failures=$((failures + 1))
}

module_name() {
  local path="${1%.lean}"
  echo "${path//\//.}"
}

check_frontier_module() {
  local file="$1" module
  module="$(module_name "$file")"
  grep -q "$AUDITED_NAMESPACE" "$file" ||
    report_failure "$file has no 'namespace HJepa.Frontier', so the audit cannot see it"
  grep -rqxF "import $module" "$FRONTIER_ROOT" "$FRONTIER_DIR" ||
    report_failure "$file is never imported, so Lean does not check it"
}

check_reviewed_library_is_independent() {
  if grep -rn '^import HJepaFrontier' HJepa.lean HJepa; then
    report_failure "the reviewed library imports an unreviewed Frontier module"
  fi
}

if [ -d "$FRONTIER_DIR" ]; then
  while IFS= read -r file; do
    check_frontier_module "$file"
  done < <(find "$FRONTIER_DIR" -name '*.lean' | sort)
fi
check_reviewed_library_is_independent

if [ "$failures" -gt 0 ]; then
  echo "$failures Frontier convention failure(s)."
  exit 1
fi
echo "PASS: Frontier conventions hold."

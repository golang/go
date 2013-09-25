#!/bin/bash
#
# Simple test of Go oracle/Emacs integration.
# Requires that GOROOT and GOPATH are set.
# Side effect: builds and installs oracle in $GOROOT.

set -eu

[ -z "$GOROOT" ] && { echo "Error: GOROOT is unset." >&2; exit 1; }
[ -z "$GOPATH" ] && { echo "Error: GOPATH is unset." >&2; exit 1; }

log=/tmp/$(basename $0)-$$.log
thisdir=$(dirname $0)

function die() {
  echo "Error: $@."
  cat $log
  exit 1
} >&2

trap "rm -f $log" EXIT

# Build and install oracle.
go get code.google.com/p/go.tools/cmd/oracle || die "'go get' failed"
mv -f $GOPATH/bin/oracle $GOROOT/bin/
$GOROOT/bin/oracle >$log 2>&1 || true # (prints usage and exits 1)
grep -q "Run.*help" $log || die "$GOROOT/bin/oracle not installed"


# Run Emacs, set the scope to the oracle tool itself,
# load ./main.go, and describe the "fmt" import.
emacs --batch --no-splash --no-window-system --no-init \
    --load $GOROOT/misc/emacs/go-mode.el \
    --load $thisdir/oracle.el \
    --eval '
(progn
  (setq go-oracle-scope "code.google.com/p/go.tools/cmd/oracle")
  (find-file "'$thisdir'/main.go")
  (search-forward "\"fmt\"")
  (backward-char)
  (go-oracle-describe)
  (princ (with-current-buffer "*go-oracle*"
                              (buffer-substring-no-properties (point-min) (point-max))))
  (kill-emacs 0))
' main.go >$log 2>&1 || die "emacs command failed"

# Check that Println is mentioned.
grep -q "fmt/print.go.*func  Println" $log || die "didn't find expected lines in log; got:"

echo "PASS"

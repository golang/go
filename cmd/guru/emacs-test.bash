#!/bin/bash
#
# Simple test of Go guru/Emacs integration.
# Requires that GOROOT and GOPATH are set.
# Side effect: builds and installs guru in $GOROOT.

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

# Build and install guru.
go get golang.org/x/tools/cmd/guru || die "'go get' failed"
mv -f $GOPATH/bin/guru $GOROOT/bin/
$GOROOT/bin/guru >$log 2>&1 || true # (prints usage and exits 1)
grep -q "Run.*help" $log || die "$GOROOT/bin/guru not installed"

# Run Emacs, set the scope to the guru tool itself,
# load ./main.go, and describe the "fmt" import.
emacs --batch --no-splash --no-window-system --no-init \
    --load $GOPATH/src/github.com/dominikh/go-mode.el/go-mode.el \
    --load $thisdir/guru.el \
    --eval '
(progn
  (princ (emacs-version)) ; requires Emacs v23
  (find-file "'$thisdir'/main.go")
  (search-forward "\"fmt\"")
  (backward-char)
  (go-guru-describe)
  (princ (with-current-buffer "*go-guru*"
                              (buffer-substring-no-properties (point-min) (point-max))))
  (kill-emacs 0))
' main.go >$log 2>&1 || die "emacs command failed"

# Check that Println is mentioned.
grep -q "fmt/print.go.*func  Println" $log || die "didn't find expected lines in log; got:"

echo "PASS"

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

# Usage: run_emacs <elisp>
function run_emacs() {
    emacs --batch --no-splash --no-window-system --no-init \
	  --load $GOPATH/src/github.com/dominikh/go-mode.el/go-mode.el \
	  --load $thisdir/guru.el \
	  --eval "$1" >$log 2>&1 || die "emacs command failed"
}

# Usage: expect_log <regex>
function expect_log() {
    grep -q "$1" $log || die "didn't find expected lines in log; got:"
}

# Load main.go and describe the "fmt" import.
# Check that Println is mentioned.
run_emacs '
(progn
  (princ (emacs-version)) ; requires Emacs v23
  (find-file "'$thisdir'/main.go")
  (insert "// modify but do not save the editor buffer\n")
  (search-forward "\"fmt\"")
  (backward-char)
  (go-guru-describe)
  (princ (with-current-buffer "*go-guru*"
                              (buffer-substring-no-properties (point-min) (point-max))))
  (kill-emacs 0))
'
expect_log "fmt/print.go.*func  Println"

# Jump to the definition of flag.Bool.
run_emacs '
(progn
  (find-file "'$thisdir'/main.go")
  (search-forward "flag.Bool")
  (backward-char)
  (go-guru-definition)
  (message "file: %s" (buffer-file-name))
  (message "line: %s" (buffer-substring (line-beginning-position)
                                        (line-end-position)))
  (kill-emacs 0))
'
expect_log "^file: .*flag.go"
expect_log "^line: func Bool"

echo "PASS"

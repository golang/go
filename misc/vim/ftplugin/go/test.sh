#!/bin/bash -e
#
# Copyright 2012 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
#
# Tests for import.vim.

cd $(dirname $0)

cat > base.go <<EOF
package test

import (
	"bytes"
	"io"
	"net"

	"mycorp/foo"
)
EOF

fail=0

# usage: test_one new_import pattern
# Pattern is a PCRE expression that will match across lines.
test_one() {
  echo 2>&1 -n "Import $1: "
  vim -e -s -u /dev/null -U /dev/null --noplugin -c "source import.vim" \
    -c "Import $1" -c 'wq! test.go' base.go
  # ensure blank lines are treated correctly
  if ! gofmt test.go | cmp test.go; then
    echo 2>&1 "gofmt conflict"
    gofmt test.go | diff -u test.go - | sed "s/^/\t/" 2>&1
    fail=1
    return
  fi
  if ! grep -P -q "(?s)$2" test.go; then
    echo 2>&1 "$2 did not match"
    cat test.go | sed "s/^/\t/" 2>&1
    fail=1
    return
  fi
  echo 2>&1 "ok"
}

test_one baz '"baz".*"bytes"'
test_one io/ioutil '"io".*"io/ioutil".*"net"'
test_one myc '"io".*"myc".*"net"'  # prefix of a site prefix
test_one nat '"io".*"nat".*"net"'
test_one net/http '"net".*"net/http".*"mycorp/foo"'
test_one zoo '"net".*"zoo".*"mycorp/foo"'
test_one mycorp/bar '"net".*"mycorp/bar".*"mycorp/foo"'
test_one mycorp/goo '"net".*"mycorp/foo".*"mycorp/goo"'

rm -f base.go test.go
if [ $fail -gt 0 ]; then
  echo 2>&1 "FAIL"
  exit 1
fi
echo 2>&1 "PASS"

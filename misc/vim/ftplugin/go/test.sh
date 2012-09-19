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

# usage: test_one command pattern
# Pattern is a PCRE expression that will match across lines.
test_one() {
  echo 2>&1 -n "$1: "
  vim -e -s -u /dev/null -U /dev/null --noplugin -c "source import.vim" \
    -c "$1" -c 'wq! test.go' base.go
  # ensure blank lines are treated correctly
  if ! gofmt test.go | cmp test.go; then
    echo 2>&1 "gofmt conflict"
    gofmt test.go | diff -u test.go - | sed "s/^/	/" 2>&1
    fail=1
    return
  fi
  if ! grep -P -q "(?s)$2" test.go; then
    echo 2>&1 "$2 did not match"
    cat test.go | sed "s/^/	/" 2>&1
    fail=1
    return
  fi
  echo 2>&1 "ok"
}

# Tests for Import

test_one "Import baz" '"baz".*"bytes"'
test_one "Import io/ioutil" '"io".*"io/ioutil".*"net"'
test_one "Import myc" '"io".*"myc".*"net"'  # prefix of a site prefix
test_one "Import nat" '"io".*"nat".*"net"'
test_one "Import net/http" '"net".*"net/http".*"mycorp/foo"'
test_one "Import zoo" '"net".*"zoo".*"mycorp/foo"'
test_one "Import mycorp/bar" '"net".*"mycorp/bar".*"mycorp/foo"'
test_one "Import mycorp/goo" '"net".*"mycorp/foo".*"mycorp/goo"'

# Tests for Drop

cat > base.go <<EOF
package test

import (
	"foo"

	"something"
	"zoo"
)
EOF

test_one "Drop something" '\([^"]*"foo"[^"]*"zoo"[^"]*\)'

rm -f base.go test.go
if [ $fail -gt 0 ]; then
  echo 2>&1 "FAIL"
  exit 1
fi
echo 2>&1 "PASS"

#!/usr/bin/env bash
# Copyright 2015 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This directory is intended to test the use of Go with sanitizers
# like msan, asan, etc.  See https://github.com/google/sanitizers .

set -e

# The sanitizers were originally developed with clang, so prefer it.
CC=cc
if test -x "$(type -p clang)"; then
  CC=clang
fi
export CC

TMPDIR=${TMPDIR:-/tmp}
echo > ${TMPDIR}/testsanitizers$$.c
if $CC -fsanitize=memory -c ${TMPDIR}/testsanitizers$$.c -o ${TMPDIR}/testsanitizers$$.o 2>&1 | grep "unrecognized" >& /dev/null; then
  echo "skipping msan test: -fsanitize=memory not supported"
  rm -f ${TMPDIR}/testsanitizers$$.*
  exit 0
fi
rm -f ${TMPDIR}/testsanitizers$$.*

# The memory sanitizer in versions of clang before 3.6 don't work with Go.
if $CC --version | grep clang >& /dev/null; then
  ver=$($CC --version | sed -e 's/.* version \([0-9.-]*\).*/\1/')
  major=$(echo $ver | sed -e 's/\([0-9]*\).*/\1/')
  minor=$(echo $ver | sed -e 's/[0-9]*\.\([0-9]*\).*/\1/')
  if test "$major" -lt 3 || test "$major" -eq 3 -a "$minor" -lt 6; then
    echo "skipping msan test; clang version $major.$minor (older than 3.6)"
    exit 0
  fi

  # Clang before 3.8 does not work with Linux at or after 4.1.
  # golang.org/issue/12898.
  if test "$major" -lt 3 || test "$major" -eq 3 -a "$minor" -lt 8; then
    if test "$(uname)" = Linux; then
      linuxver=$(uname -r)
      linuxmajor=$(echo $linuxver | sed -e 's/\([0-9]*\).*/\1/')
      linuxminor=$(echo $linuxver | sed -e 's/[0-9]*\.\([0-9]*\).*/\1/')
      if test "$linuxmajor" -gt 4 || test "$linuxmajor" -eq 4 -a "$linuxminor" -ge 1; then
        echo "skipping msan test; clang version $major.$minor (older than 3.8) incompatible with linux version $linuxmajor.$linuxminor (4.1 or newer)"
        exit 0
      fi
    fi
  fi
fi

status=0

if ! go build -msan std; then
  echo "FAIL: build -msan std"
  status=1
fi

if ! go run -msan msan.go; then
  echo "FAIL: msan"
  status=1
fi

if ! CGO_LDFLAGS="-fsanitize=memory" CGO_CPPFLAGS="-fsanitize=memory" go run -msan -a msan2.go; then
  echo "FAIL: msan2 with -fsanitize=memory"
  status=1
fi

if ! go run -msan -a msan2.go; then
  echo "FAIL: msan2"
  status=1
fi

if ! go run -msan msan3.go; then
  echo "FAIL: msan3"
  status=1
fi

if ! go run -msan msan4.go; then
  echo "FAIL: msan4"
  status=1
fi

if go run -msan msan_fail.go 2>/dev/null; then
  echo "FAIL: msan_fail"
  status=1
fi

exit $status

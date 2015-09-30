#!/usr/bin/env bash
# Copyright 2015 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This directory is intended to test the use of Go with sanitizers
# like msan, asan, etc.  See https://github.com/google/sanitizers .

set -e

# The sanitizers were originally developed with clang, so prefer it.
CC=cc
if test "$(type -p clang)" != ""; then
  CC=clang
fi
export CC

if $CC -fsanitize=memory 2>&1 | grep "unrecognized" >& /dev/null; then
  echo "skipping msan test: -fsanitize=memory not supported"
else
  go run msan.go
fi

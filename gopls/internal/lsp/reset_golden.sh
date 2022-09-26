#!/bin/bash
#
# Copyright 2022 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
#
# Updates the *.golden files ... to match the tests' current behavior.

set -eu

GO117BIN="go1.17.9"

command -v $GO117BIN >/dev/null 2>&1 || {
  go install golang.org/dl/$GO117BIN@latest
  $GO117BIN download
}

find ./internal/lsp/testdata -name *.golden ! -name summary*.txt.golden -delete
# Here we intentionally do not run the ./internal/lsp/source tests with
# -golden. Eventually these tests will be deleted, and in the meantime they are
# redundant with the ./internal/lsp tests.
#
# Note: go1.17.9 tests must be run *before* go tests, as by convention the
# golden output should match the output of gopls built with the most recent
# version of Go. If output differs at 1.17, tests must be tolerant of the 1.17
# output.
$GO117BIN test ./internal/lsp -golden
go test ./internal/lsp -golden
$GO117BIN test ./test  -golden
go test ./test  -golden

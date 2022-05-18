#!/bin/bash -eu

# Copyright 2020 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -o pipefail

# Copy golang.org/x/vuln/cmd/govulncheck/internal/govulncheck into this directory.
# Assume the x/vuln repo is a sibling of the tools repo.

rm -f *.go
cp ../../../../vuln/cmd/govulncheck/internal/govulncheck/*.go .

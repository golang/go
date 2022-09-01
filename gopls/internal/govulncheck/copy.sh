#!/bin/bash -eu

# Copyright 2020 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -o pipefail

# Copy golang.org/x/vuln/cmd/govulncheck/internal/govulncheck into this directory.
# Assume the x/vuln repo is a sibling of the tools repo.

rm -f *.go
cp ../../../../vuln/cmd/govulncheck/internal/govulncheck/*.go .

sed -i '' 's/\"golang.org\/x\/vuln\/internal\/semver\"/\"golang.org\/x\/tools\/gopls\/internal\/govulncheck\/semver\"/g' *.go
sed -i '' -e '4 i\
' -e '4 i\
//go:build go1.18' -e '4 i\
// +build go1.18' *.go

# Copy golang.org/x/vuln/internal/semver that
# golang.org/x/vuln/cmd/govulncheck/internal/govulncheck
# depends on.

mkdir -p semver
cd semver
rm -f *.go
cp ../../../../../vuln/internal/semver/*.go .
sed -i '' -e '4 i\
' -e '4 i\
//go:build go1.18' -e '4 i\
// +build go1.18' *.go

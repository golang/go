#!/usr/bin/env bash
# Copyright 2020 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# A quick and dirty way to obtain code coverage from rulegen's main func. For
# example:
#
#     ./cover.bash && go tool cover -html=cover.out
#
# This script is needed to set up a temporary test file, so that we don't break
# regular 'go run .' usage to run the generator.

cat >main_test.go <<-EOF
	//go:build ignore

	package main

	import "testing"

	func TestCoverage(t *testing.T) { main() }
EOF

go test -run='^TestCoverage$' -coverprofile=cover.out "$@" *.go

rm -f main_test.go

#!/usr/bin/env bash

# Copyright 2015 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Run this script to obtain an up-to-date vendored version of math/big.

BIGDIR=../../../../math/big

# Start from scratch.
rm *.go

# We don't want any assembly files.
cp $BIGDIR/*.go .

# Use pure Go arith ops w/o build tag.
sed 's/^\/\/ \+build math_big_pure_go$//' arith_decl_pure.go > arith_decl.go
rm arith_decl_pure.go

# gofmt to clean up after sed
gofmt -w .

# Test that it works
go test -short

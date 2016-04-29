#!/usr/bin/env bash

# Copyright 2016 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Run this script to obtain an up-to-date vendored version of debug/pe

PEDIR=../../../../debug/pe

# Start from scratch.
rm testdata/*
rm *.go
rmdir testdata

# Copy all files.
mkdir testdata
cp $PEDIR/*.go .
cp $PEDIR/testdata/* ./testdata

# go1.4 (bootstrap) does not know what io.SeekStart is.
sed -i 's|io.SeekStart|os.SEEK_SET|' *.go

# goimports to clean up after sed
goimports -w .

# gofmt to clean up after sed
gofmt -w .

# Test that it works
go test -short

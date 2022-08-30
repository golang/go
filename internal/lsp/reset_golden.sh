#!/bin/bash
#
# Copyright 2022 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
#
# Regenerates the *.golden files in the ./internal/lsp/testdata directory.

set -eu

find ./internal/lsp/testdata -name *.golden ! -name summary*.txt.golden -delete
go test ./internal/lsp/source  -golden
go test ./internal/lsp/ -golden
go test ./internal/lsp/cmd  -golden

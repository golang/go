#!/bin/bash
#
# Copyright 2022 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
#
# Creates a zip file containing all numbered versions
# of the commit history of a large source file, for use
# as input data for the tests of the diff algorithm.
#
# Run script from root of the x/tools repo.

set -eu

# WARNING: This script will install the latest version of $file
# The largest real source file in the x/tools repo.
# file=internal/lsp/source/completion/completion.go
# file=internal/lsp/source/diagnostics.go
file=internal/lsp/protocol/tsprotocol.go

tmp=$(mktemp -d)
git log $file |
  awk '/^commit / {print $2}' |
  nl -ba -nrz |
  while read n hash; do
    git checkout --quiet $hash $file
    cp -f $file $tmp/$n
  done
(cd $tmp && zip -q - *) > testdata.zip
rm -fr $tmp
git restore --staged $file
git restore $file
echo "Created testdata.zip"

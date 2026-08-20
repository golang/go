#!/bin/bash

# Copyright 2026 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.


# This is based on the instructions at https://github.com/intelxed/xed

set -e

XEDTAG=v2025.03.02
MBUILDTAG=v2024.11.04
DATA=./extern

trace() {
    set -x
    "$@"
    { local rc=$?; set +x; } 2>/dev/null
    return $rc
}

echo "This will download xed to $DATA/xed and $DATA/mbuild and compile it."
echo "It requires Python, a C toolchain, and probably other things."
read -p "Continue (y/n)? " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

if [[ -d "$DATA/xed" ]]; then
    echo 2>&1 "$DATA/xed already downloaded"
else
    trace git clone -b $XEDTAG https://github.com/intelxed/xed.git $DATA/xed
fi
if [[ -d "$DATA/mbuild" ]]; then
    echo 2>&1 "$DATA/mbuild already downloaded"
else
    trace git clone -b $MBUILDTAG https://github.com/intelxed/mbuild.git $DATA/mbuild
fi

if [[ -f $DATA/xed/obj/dgen/all-dec-instructions.txt ]]; then
    echo 2>&1 "XED data already compiled"
else
    trace cd $DATA/xed
    trace ./mfile.py
fi

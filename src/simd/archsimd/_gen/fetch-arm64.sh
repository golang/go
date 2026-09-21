#!/bin/bash

# Copyright 2026 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

TAR=ISA_A64_xml_A_profile-2026-03_96
DIR=ISA_A64_xml_A_profile_2026-03_96-2026-03_rel
DATA=./extern

trace() {
    set -x
    "$@"
    { local rc=$?; set +x; } 2>/dev/null
    return $rc
}

if [[ -d "$DATA/$DIR" ]]; then
    echo 2>&1 "ISA description already downloaded"
else
    tartmp=$(mktemp tmp.XXXXXXXXXX.tar.gz)
    trace curl -o "$tartmp" https://developer.arm.com/-/cdn-downloads/permalink/Exploration-Tools-A64-ISA/ISA_A64/$TAR.tar.gz
    # Check that it has the expected path
    want="$DIR/abs_advsimd.xml"
    if ! tar tzf "$tartmp" | grep -qxF "$want"; then
        echo 2>&1 "Archive $tartmp does not contain expected file $want"
        exit 1
    fi
    # This tar file has multiple top-level directories and files. We want just $DIR
    trace mkdir -p "$DATA"
    trace tar -xz -C "$DATA" -f "$tartmp" "$DIR"
    trace rm $tartmp
fi

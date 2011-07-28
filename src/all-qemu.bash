#!/usr/bin/env bash
# Copyright 2011 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Run all.bash but exclude tests that depend on functionality
# missing in QEMU's system call emulation.

export GOARCH=arm
export NOTEST=""
NOTEST="$NOTEST big" # just slow
NOTEST="$NOTEST go/build" # wants to run cgo
NOTEST="$NOTEST http http/cgi net rpc syslog websocket"  # no localhost network
NOTEST="$NOTEST os"  # 64-bit seek fails

./all.bash

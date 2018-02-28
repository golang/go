#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

if [ ! -f run.bash ]; then
	echo 'clean.bash must be run from $GOROOT/src' 1>&2
	exit 1
fi
export GOROOT="$(cd .. && pwd)"

gobin="${GOBIN:-../bin}"
if ! "$gobin"/go help >/dev/null 2>&1; then
	echo 'cannot find go command; nothing to clean' >&2
	exit 1
fi

"$gobin/go" clean -i std
"$gobin/go" tool dist clean
"$gobin/go" clean -i cmd

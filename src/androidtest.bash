#!/usr/bin/env bash
# Copyright 2014 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# For testing Android.

set -e
ulimit -c 0 # no core files

if [ ! -f make.bash ]; then
	echo 'androidtest.bash must be run from $GOROOT/src' 1>&2
	exit 1
fi

if [ -z $GOOS ]; then
	export GOOS=android
fi
if [ "$GOOS" != "android" ]; then
	echo "androidtest.bash requires GOOS=android, got GOOS=$GOOS" 1>&2
	exit 1
fi

if [ -n "$GOARM" ] && [ "$GOARM" != "7" ]; then
	echo "android only supports GOARM=7, got GOARM=$GOARM" 1>&2
	exit 1
fi

export CGO_ENABLED=1
unset GOBIN

export GOROOT=$(dirname $(pwd))
# Put the exec wrapper into PATH
export PATH=$GOROOT/bin:$PATH

# Run standard tests.
bash all.bash

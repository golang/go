#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# GORUNFLAGS can be passed as an environment variable
# in order to pass optional flags through to the
# run.bash invocation.

set -e
if [ ! -f make.bash ]; then
	echo 'all.bash must be run from $GOROOT/src' 1>&2
	exit 1
fi
OLDPATH="$PATH"
. ./make.bash "$@" --no-banner
bash run.bash --no-rebuild ${GORUNFLAGS}
PATH="$OLDPATH"
$GOTOOLDIR/dist banner  # print build info

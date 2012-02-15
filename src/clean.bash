#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

eval $(go tool dist env)

if [ ! -x $GOTOOLDIR/dist ]; then
	echo 'cannot find $GOTOOLDIR/dist; nothing to clean' >&2
	exit 1
fi

"$GOBIN/go" clean -i std
$GOTOOLDIR/dist clean

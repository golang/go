#!/bin/bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# TODO(rsc): delete in favor of all.bash once nacl support is complete

set -e
bash make.bash

xcd() {
	echo
	echo --- cd $1
	builtin cd $1
}

(xcd ../test
./run-nacl
) || exit $?

#!/bin/bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e
bash make-arm.bash

# TODO(kaib): add in proper tests
#bash run.bash

set -e

xcd() {
	echo
	echo --- cd $1
	builtin cd $1
}

(xcd ../test
./run-arm
) || exit $?

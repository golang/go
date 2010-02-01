#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# TODO(rsc): delete in favor of all.bash once nacl support is complete

export GOARCH=386
export GOOS=nacl

set -e
bash make.bash

xcd() {
	echo
	echo --- cd $1
	builtin cd $1
}

(xcd pkg/exp/nacl/srpc
make clean
make install
) || exit $?

(xcd pkg/exp/nacl/av
make clean
make install
) || exit $?

(xcd pkg/exp/4s
make clean
make
) || exit $?

(xcd pkg/exp/spacewar
make clean
make
) || exit $?

(xcd ../test
./run-nacl
) || exit $?

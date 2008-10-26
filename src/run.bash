#!/bin/bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

xcd() {
	echo
	echo --- cd $1
	builtin cd $1
}

(xcd lib/reflect
make clean
time make
bash test.bash
)

(xcd lib/regexp
make clean
time make
make test
)

(xcd ../usr/gri/gosrc
make clean
time make
# make test
)

(xcd ../usr/gri/pretty
make clean
time make
make test
)


(xcd ../test
./run
)


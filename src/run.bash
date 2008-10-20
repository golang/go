#!/bin/bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

xcd() {
	builtin cd $1
	echo --- cd $1
}

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


(xcd ../usr/r/refl
rm -f *.6 6.out
6g refl.go
6g printf.go
6g main.go
6l main.6
6.out
)

(xcd ../test
./run
)


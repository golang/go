#!/bin/bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

bash clean.bash

case "$GOARCH" in
386)	O=8;;
amd64)	O=6;;
arm)	O=5;;
*)
	echo 'unknown $GOARCH' 1>&2
	exit 1
esac

cd ${O}l
bash mkenam
make enam.o
cd ..

for i in cc ${O}l ${O}a ${O}c gc ${O}g ar db nm acid cov godefs prof gotest
do
	echo; echo; echo %%%% making $i %%%%; echo
	cd $i
	make install
	cd ..
done

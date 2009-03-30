#!/bin/bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

bash clean.bash

cd 6l
bash mkenam
make enam.o
cd ..

for i in cc 6l 6a 6c gc 6g ar db nm acid cov gobuild godefs prof gotest
do
	echo; echo; echo %%%% making $i %%%%; echo
	cd $i
	make install
	cd ..
done

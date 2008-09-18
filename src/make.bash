#!/bin/bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e
export MAKEFLAGS=-j4

bash clean.bash

for i in lib9 libbio libmach_amd64 libregexp
do
	cd $i
	make install
	cd ..
done

for i in cmd runtime
do
	cd $i
	bash make.bash
	cd ..
done

# do these after go compiler and runtime are built
for i in syscall
do
	echo; echo; echo %%%% making $i %%%%; echo
	cd $i
	make install
	cd ..
done

for i in lib
do
	cd $i
	bash make.bash
	cd ..
done

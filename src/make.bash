#!/bin/bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e
export MAKEFLAGS=-j4

bash clean.bash

for i in lib9 libbio libmach_amd64 libregexp cmd runtime syscall lib
do
	echo; echo; echo %%%% making $i %%%%; echo
	cd $i
	case $i in
	cmd | lib)
		bash make.bash
		;;
	*)
		make install
	esac
	cd ..
done


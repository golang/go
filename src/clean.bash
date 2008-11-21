#!/bin/bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

for i in lib9 libbio libmach_amd64 libregexp cmd runtime lib
do
	cd $i
	case $i in
	cmd)
		bash clean.bash
		;;
	*)
		make clean
	esac
	cd ..
done

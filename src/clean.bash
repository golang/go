#!/bin/bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

rm -rf $GOROOT/pkg/[0-9a-zA-Z_]*
rm -f $GOROOT/lib/*.[6a]
for i in lib9 libbio libmach libregexp cmd pkg
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

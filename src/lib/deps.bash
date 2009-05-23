#!/bin/bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

OUT="Make.deps"
TMP="Make.deps.tmp"

# Get list of directories from Makefile
dirs=$(sed '1,/^DIRS=/d; /^$/,$d; s/\\//g' Makefile)
dirpat=$(echo $dirs | sed 's/ /|/g; s/.*/^(&)$/')

for dir in $dirs; do (
	cd $dir || exit 1

	sources=$(sed -n 's/\.\$O\\/.go/p' Makefile)
	sources=$(ls $sources 2> /dev/null)  # remove .s, .c, etc.

	deps=$(
		sed -n '/^import.*"/p; /^import[ \t]*(/,/^)/p' $sources /dev/null |
		cut -d '"' -f2 |
		egrep "$dirpat" |
		grep -v "^$dir\$" |
		sed 's/$/.install/' |
		sort -u
	)

	echo $dir.install: $deps
) done > $TMP

p4 open $OUT
mv $TMP $OUT

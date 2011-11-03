#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

eval $(gomake --no-print-directory -f ../Make.inc go-env)

OUT="Make.deps"
TMP="Make.deps.tmp"

if [ -f $OUT ] && ! [ -w $OUT ]; then
	echo "$0: $OUT is read-only; aborting." 1>&2
	exit 1
fi

# Get list of directories from Makefile
dirs=$(gomake --no-print-directory echo-dirs)
dirpat=$(echo $dirs C | awk '{
	for(i=1;i<=NF;i++){ 
		x=$i
		gsub("/", "\\/", x)
		printf("/^(%s)$/\n", x)
	}
}'
	# Add packages' old names. TODO: clean up after renaming.
	echo "/^(asn1)$/"
	echo "/^(csv)$/"
	echo "/^(gob)$/"
	echo "/^(json)$/"
	echo "/^(xml)$/"
)

for dir in $dirs; do (
	cd $dir >/dev/null || exit 1

	sources=$(sed -n 's/^[ 	]*\([^ 	]*\.go\)[ 	]*\\*[ 	]*$/\1/p' Makefile)
	sources=$(echo $sources | sed 's/\$(GOOS)/'$GOOS'/g')
	sources=$(echo $sources | sed 's/\$(GOARCH)/'$GOARCH'/g')
	# /dev/null here means we get an empty dependency list if $sources is empty
	# instead of listing every file in the directory.
	sources=$(ls $sources /dev/null 2> /dev/null)  # remove .s, .c, etc.

	deps=$(
		sed -n '/^import.*"/p; /^import[ \t]*(/,/^)/p' $sources /dev/null |
		cut -d '"' -f2 |
		awk "$dirpat" |
		grep -v "^$dir\$" |
		sed 's/$/.install/' |
		# TODO: rename the dependencies for renamed directories.  TODO: clean up after renaming.
		# awk is overkill but it's easy to understand.
		awk '
			/^asn1.install$/ { print "encoding/asn1.install"; next }
			/^csv.install$/ { print "encoding/csv.install"; next }
			/^gob.install$/ { print "encoding/gob.install"; next }
			/^json.install$/ { print "encoding/json.install"; next }
			/^xml.install$/ { print "encoding/xml.install"; next }
			{print}
		' |
		sed 's;^C\.install;runtime/cgo.install;' |
		sort -u
	)

	echo $dir.install: $deps
) done > $TMP

mv $TMP $OUT

if (egrep -v '^(exp|old)/' $OUT | egrep -q " (exp|old)/"); then
	echo "$0: $OUT contains dependencies to exp or old packages"
        exit 1
fi

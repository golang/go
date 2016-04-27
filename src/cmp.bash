#!/usr/bin/env bash

# Copyright 2016 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# A simple script to compare differences between
# assembly listings for packages built with different
# compiler flags. It is useful to inspect the impact
# of a compiler change across all std lib packages.
#
# The script builds the std library (make.bash) once
# with FLAGS1 and once with FLAGS2 and compares the
# "go build <pkg>" assembly output for each package
# and lists the packages with differences.
#
# It leaves and old.txt and new.txt file in the package
# directories for the packages with differences.

FLAGS1="-newexport=0"
FLAGS2="-newexport=1"

echo
echo
echo "1a) clean build using $FLAGS1"
(export GO_GCFLAGS="$FLAGS1"; sh make.bash)

echo
echo
echo "1b) save go build output for all packages"
for pkg in `go list std`; do
	echo $pkg
	DIR=$GOROOT/src/$pkg
	go build -gcflags "$FLAGS1 -S" -o /dev/null $pkg &> $DIR/old.txt
done

echo
echo
echo "2a) clean build using $FLAGS2"
(export GO_GCFLAGS="$FLAGS2"; sh make.bash)

echo
echo
echo "2b) save go build output for all packages"
for pkg in `go list std`; do
	echo $pkg
	DIR=$GOROOT/src/$pkg
	go build -gcflags "$FLAGS2 -S" -o /dev/null $pkg &> $DIR/new.txt
done

echo
echo
echo "3) compare assembly files"
for pkg in `go list std`; do
	DIR=$GOROOT/src/$pkg

	if cmp $DIR/old.txt $DIR/new.txt &> /dev/null
	then rm $DIR/old.txt $DIR/new.txt
	else echo "==> $DIR"
	fi
done

# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

#!/bin/bash

function buildfiles() {
	rm -f *.6
	for i
	do
		base=$(basename $i .go)
		echo 6g -o $GOROOT/pkg/$base.6 $i
		6g -o $GOROOT/pkg/$base.6 $i
	done
}

function builddirs() {
	for i
	do
		echo; echo; echo %%%% making lib/$i %%%%; echo
		(cd $i; make install)
	done
}

set -e
rm -f *.6

# Don't sort the elements of the lists - some of the orderings matter.

buildfiles	strings.go

builddirs	syscall\
		math\
		os\
		strconv\
		container/array\
		reflect\
	
buildfiles	io.go

builddirs	fmt

buildfiles	flag.go\
		container/vector.go\
		rand.go\
		sort.go\
		bufio.go\
		once.go\
		bignum.go\
		testing.go\
	
builddirs	net\
		time\
		http\
		regexp\

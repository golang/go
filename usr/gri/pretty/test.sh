# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

#!/bin/bash

TMP1=test_tmp1.go
TMP2=test_tmp2.go
COUNT=0

count() {
	let COUNT=$COUNT+1
	let M=$COUNT%10
	if [ $M == 0 ]; then
		echo -n "."
	fi
}


# apply to one file
apply1() {
	#echo $1 $2
	case `basename $F` in
	selftest1.go | func3.go | bug014.go | bug029.go | bug032.go | bug050.go | \
	bug068.go | bug088.go | bug083.go | bug106.go ) ;;  # skip - files contain syntax errors
	* ) $1 $2; count ;;
	esac
}


# apply to local files
applydot() {
	for F in *.go
	do
		apply1 $1 $F
	done
}


# apply to all files in the list
apply() {
	for F in \
		$GOROOT/usr/gri/pretty/*.go \
		$GOROOT/test/*.go \
		$GOROOT/test/bugs/*.go \
		$GOROOT/test/fixedbugs/*.go \
		$GOROOT/src/lib/*.go \
		$GOROOT/src/lib/*/*.go \
		$GOROOT/usr/r/*/*.go
	do
		apply1 $1 $F
	done
}


cleanup() {
	rm -f $TMP1 $TMP2
}


silent() {
	cleanup
	./pretty -s $1 > $TMP1
	if [ $? != 0 ]; then
		cat $TMP1
		echo "Error (silent mode test): test.sh $1"
		exit 1
	fi
}


idempotent() {
	cleanup
	./pretty $1 > $TMP1
	./pretty $TMP1 > $TMP2
	cmp -s $TMP1 $TMP2
	if [ $? != 0 ]; then
		diff $TMP1 $TMP2
		echo "Error (idempotency test): test.sh $1"
		exit 1
	fi
}


valid() {
	cleanup
	./pretty $1 > $TMP1
	6g -o /dev/null $TMP1
	if [ $? != 0 ]; then
		echo "Error (validity test): test.sh $1"
		exit 1
	fi
}


runtest() {
	#echo "Testing silent mode"
	cleanup
	$1 silent $2

	#echo "Testing idempotency"
	cleanup
	$1 idempotent $2
}


runtests() {
	if [ $# == 0 ]; then
		runtest apply
		# verify the pretty-printed files can be compiled with 6g again
		# do it in local directory only because of the prerequisites required
		#echo "Testing validity"
		cleanup
		applydot valid
	else
		for F in $*; do
			runtest apply1 $F
		done
	fi
}


# run selftest1 always
./pretty -t selftest1.go > $TMP1
if [ $? != 0 ]; then
	cat $TMP1
	echo "Error (selftest1): pretty -t selftest1.go"
	exit 1
fi
count


# run over all .go files
runtests $*
cleanup

# done
echo
echo "PASSED ($COUNT tests)"

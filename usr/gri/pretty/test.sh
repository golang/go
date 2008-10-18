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


apply1() {
	#echo $1 $2
	$1 $2
	count
}


apply() {
	for F in \
		$GOROOT/usr/gri/pretty/*.go \
		$GOROOT/usr/gri/gosrc/*.go \
		$GOROOT/test/*.go \
		$GOROOT/src/pkg/*.go \
		$GOROOT/src/lib/*.go \
		$GOROOT/src/lib/*/*.go \
		$GOROOT/usr/r/*/*.go
	do
		case `basename $F` in
		selftest.go | func3.go ) ;; # skip - these are test cases for syntax errors
		* ) apply1 $1 $F ;;
		esac
	done
}


cleanup() {
	rm -f $TMP1 $TMP2
}


silent() {
	cleanup
	pretty -s $1 > $TMP1
	if [ $? != 0 ]; then
		cat $TMP1
		echo "Error (silent mode test): test.sh $1"
		exit 1
	fi
}


idempotent() {
	cleanup
	pretty $1 > $TMP1
	pretty $TMP1 > $TMP2
	cmp -s $TMP1 $TMP2
	if [ $? != 0 ]; then
		diff $TMP1 $TMP2
		echo "Error (idempotency test): test.sh $1"
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
	else
		for F in $*; do
			runtest apply1 $F
		done
	fi
}


# run selftest always
pretty -t selftest.go > $TMP1
if [ $? != 0 ]; then
	cat $TMP1
	echo "Error (selftest): pretty -t selftest.go"
	exit 1
fi
count


# run over all .go files
runtests $*
cleanup

# done
echo
echo "PASSED ($COUNT tests)"

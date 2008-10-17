# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

#!/bin/bash

TMP1=test_tmp1.go
TMP2=test_tmp2.go
COUNT=0

apply1() {
	#echo $1 $2
	$1 $2
	let COUNT=$COUNT+1
}

apply() {
	for F in \
		$GOROOT/usr/gri/pretty/*.go \
		$GOROOT/usr/gri/gosrc/*.go \
		$GOROOT/test/235.go \
		$GOROOT/test/args.go \
		$GOROOT/test/bufiolib.go \
		$GOROOT/test/char_lit.go \
		$GOROOT/test/complit.go \
		$GOROOT/test/const.go \
		$GOROOT/test/dialgoogle.go \
		$GOROOT/test/empty.go \
		$GOROOT/test/env.go \
		$GOROOT/test/float_lit.go \
		$GOROOT/test/fmt_test.go \
		$GOROOT/test/for.go \
		$GOROOT/test/func.go \
		$GOROOT/test/func1.go \
		$GOROOT/test/func2.go \
		$GOROOT/src/pkg/*.go \
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

runtests $*
cleanup
let COUNT=$COUNT/2  # divide by number of tests in runtest
echo "PASSED ($COUNT files)"


#!/usr/bin/env bash
# Copyright 2010 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

CMD="../gofmt"
TMP=test_tmp.go
COUNT=0


cleanup() {
	rm -f $TMP
}


error() {
	echo $1
	exit 1
}


count() {
	#echo $1
	let COUNT=$COUNT+1
	let M=$COUNT%10
	if [ $M == 0 ]; then
		echo -n "."
	fi
}


test() {
	count $1

	# compare against .golden file
	cleanup
	$CMD -s $1 > $TMP
	cmp -s $TMP $2
	if [ $? != 0 ]; then
		diff $TMP $2
		error "Error: simplified $1 does not match $2"
	fi

	# make sure .golden is idempotent
	cleanup
	$CMD -s $2 > $TMP
	cmp -s $TMP $2
	if [ $? != 0 ]; then
		diff $TMP $2
		error "Error: $2 is not idempotent"
	fi
}


runtests() {
	smoketest=../../../pkg/go/parser/parser.go
	test $smoketest $smoketest
	test composites.input composites.golden
	# add more test cases here
}


runtests
cleanup
echo "PASSED ($COUNT tests)"

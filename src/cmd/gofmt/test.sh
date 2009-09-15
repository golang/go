#!/bin/bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

. $GOROOT/src/Make.$GOARCH
if [ -z "$O" ]; then
	echo 'missing $O - maybe no Make.$GOARCH?' 1>&2
	exit 1
fi

CMD="./gofmt"
TMP1=test_tmp1.go
TMP2=test_tmp2.go
TMP3=test_tmp3.go
COUNT=0

count() {
	#echo $1
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
	# files with errors (skip them)
	# the following have semantic errors:
	#   bug039.go | bug040.go
	# the following are not idempotent at the moment because of comment formatting:
	# TODO: restructure script so these files are only excluded from idempotency testing
	comment.go | net.go | powser1.go | powser2.go | bug052.go | simpbool.go | "shift.go" | range.go | \
	goyacc.go | godoc.go | rpc.go | struct.go | log.go | decimal.go | tabwriter.go | encoder.go | debug.go | \
	elf.go | meteor-contest.go | elffmt.go | xml.go | \
	\
	test_errors.go | calc.go | method1.go | selftest1.go | func3.go | const2.go | \
	bug014.go | bug025.go | bug029.go | bug032.go | bug039.go | bug040.go | bug050.go |  bug068.go | \
	bug088.go | bug083.go | bug106.go | bug121.go | bug125.go | bug126.go | bug132.go | bug133.go | \
	bug134.go | bug160.go | bug163.go | bug166.go | bug169.go ) ;;
	* ) $1 $2; count $F;;
	esac
}


# apply to local files
applydot() {
	for F in `find . -name "*.go" | grep -v "OLD" | grep -v "._"`; do
		apply1 $1 $F
	done
}


# apply to all .go files we can find
apply() {
	for F in `find $GOROOT -name "*.go" | grep -v "OLD" | grep -v "._"`; do
		apply1 $1 $F
	done
}


cleanup() {
	rm -f $TMP1 $TMP2 $TMP3
}


silent() {
	cleanup
	$CMD -s $1 > $TMP1
	if [ $? != 0 ]; then
		cat $TMP1
		echo "Error (silent mode test): test.sh $1"
		exit 1
	fi
}


idempotent() {
	cleanup
	$CMD $1 > $TMP1
	if [ $? != 0 ]; then
		echo "Error (step 1 of idempotency test): test.sh $1"
		exit 1
	fi

	$CMD $TMP1 > $TMP2
	if [ $? != 0 ]; then
		echo "Error (step 2 of idempotency test): test.sh $1"
		exit 1
	fi

	$CMD $TMP2 > $TMP3
	if [ $? != 0 ]; then
		echo "Error (step 3 of idempotency test): test.sh $1"
		exit 1
	fi

	cmp -s $TMP2 $TMP3
	if [ $? != 0 ]; then
		diff $TMP2 $TMP3
		echo "Error (step 4 of idempotency test): test.sh $1"
		exit 1
	fi
}


valid() {
	cleanup
	$CMD $1 > $TMP1
	if [ $? != 0 ]; then
		echo "Error (step 1 of validity test): test.sh $1"
		exit 1
	fi

	$GC -o /dev/null $TMP1
	if [ $? != 0 ]; then
		echo "Error (step 2 of validity test): test.sh $1"
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
		# verify the pretty-printed files can be compiled with $GC again
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


# run over all .go files
runtests $*
cleanup

# done
echo
echo "PASSED ($COUNT tests)"

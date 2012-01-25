#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

eval $(gomake --no-print-directory -f ../../Make.inc go-env)
if [ -z "$O" ]; then
	echo 'missing $O - maybe no Make.$GOARCH?' 1>&2
	exit 1
fi

CMD="./gofmt"
TMP1=test_tmp1.go
TMP2=test_tmp2.go
TMP3=test_tmp3.go
COUNT=0
rm -f _failed

count() {
	#echo $1
	let COUNT=$COUNT+1
	let M=$COUNT%10
	if [ $M == 0 ]; then
		echo -n "."
	fi
}


error() {
	echo $1
	touch _failed
}

# apply to one file
apply1() {
	# the following files are skipped because they are test cases
	# for syntax errors and thus won't parse in the first place:
	case `basename "$F"` in
	func3.go | const2.go | char_lit1.go | blank1.go | ddd1.go | \
	bug014.go | bug050.go |  bug068.go |  bug083.go | bug088.go | \
	bug106.go | bug121.go | bug125.go | bug133.go | bug160.go | \
	bug163.go | bug166.go | bug169.go | bug217.go | bug222.go | \
	bug226.go | bug228.go | bug248.go | bug274.go | bug280.go | \
	bug282.go | bug287.go | bug298.go | bug299.go | bug300.go | \
	bug302.go | bug306.go | bug322.go | bug324.go | bug335.go | \
	bug340.go | bug349.go | bug351.go | bug358.go | bug367.go | \
	bug388.go | bug394.go ) return ;;
	esac
	# the following directories are skipped because they contain test
	# cases for syntax errors and thus won't parse in the first place:
	case `dirname "$F"` in
	$GOROOT/test/syntax ) return ;;
	esac
	#echo $1 $2
	"$1" "$2"; count "$F"
}


# apply to local files
applydot() {
	for F in `find . -name "*.go" | grep -v "._"`; do
		apply1 "$1" $F
	done
}


# apply to all .go files we can find
apply() {
	for F in `find "$GOROOT" -name "*.go" | grep -v "._"`; do
		apply1 "$1" $F
	done
}


cleanup() {
	rm -f $TMP1 $TMP2 $TMP3
}


silent() {
	cleanup
	$CMD "$1" > /dev/null 2> $TMP1
	if [ $? != 0 ]; then
		cat $TMP1
		error "Error (silent mode test): test.sh $1"
	fi
}


idempotent() {
	cleanup
	$CMD "$1" > $TMP1
	if [ $? != 0 ]; then
		error "Error (step 1 of idempotency test): test.sh $1"
	fi

	$CMD $TMP1 > $TMP2
	if [ $? != 0 ]; then
		error "Error (step 2 of idempotency test): test.sh $1"
	fi

	$CMD $TMP2 > $TMP3
	if [ $? != 0 ]; then
		error "Error (step 3 of idempotency test): test.sh $1"
	fi

	cmp -s $TMP2 $TMP3
	if [ $? != 0 ]; then
		diff $TMP2 $TMP3
		error "Error (step 4 of idempotency test): test.sh $1"
	fi
}


valid() {
	cleanup
	$CMD "$1" > $TMP1
	if [ $? != 0 ]; then
		error "Error (step 1 of validity test): test.sh $1"
	fi

	$GC -o /dev/null $TMP1
	if [ $? != 0 ]; then
		error "Error (step 2 of validity test): test.sh $1"
	fi
}


runtest() {
	#echo "Testing silent mode"
	cleanup
	"$1" silent "$2"

	#echo "Testing idempotency"
	cleanup
	"$1" idempotent "$2"
}


runtests() {
	if [ $# = 0 ]; then
		runtest apply
		# verify the pretty-printed files can be compiled with $GC again
		# do it in local directory only because of the prerequisites required
		#echo "Testing validity"
		# Disabled for now due to dependency problems
		# cleanup
		# applydot valid
	else
		for F in "$@"; do
			runtest apply1 "$F"
		done
	fi
}


# run over all .go files
runtests "$@"
cleanup

if [ -f _failed ]; then
	rm _failed
	exit 1
fi

# done
echo
echo "PASSED ($COUNT tests)"

#!/usr/bin/env bash

# Copyright 2013 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

check() {
	file=$1
	line=$(grep -n 'ERROR HERE' $file | sed 's/:.*//')
	if [ "$line" = "" ]; then
		echo 1>&2 misc/cgo/errors/test.bash: BUG: cannot find ERROR HERE in $file
		exit 1
	fi
	expect $file $file:$line:
}

expect() {
	file=$1
	shift
	if go build $file >errs 2>&1; then
		echo 1>&2 misc/cgo/errors/test.bash: BUG: expected cgo to fail on $file but it succeeded
		exit 1
	fi
	if ! test -s errs; then
		echo 1>&2 misc/cgo/errors/test.bash: BUG: expected error output for $file but saw none
		exit 1
	fi
	for error; do
		if ! fgrep $error errs >/dev/null 2>&1; then
			echo 1>&2 misc/cgo/errors/test.bash: BUG: expected error output for $file to contain \"$error\" but saw:
			cat 1>&2 errs
			exit 1
		fi
	done
}

check err1.go
check err2.go
check err3.go
check issue7757.go
check issue8442.go
check issue11097a.go
check issue11097b.go
expect issue13129.go C.ushort
check issue13423.go
expect issue13635.go C.uchar C.schar C.ushort C.uint C.ulong C.longlong C.ulonglong C.complexfloat C.complexdouble
check issue13830.go
check issue16116.go

if ! go build issue14669.go; then
	exit 1
fi
if ! CGO_CFLAGS="-O" go build issue14669.go; then
	exit 1
fi

if ! go run ptr.go; then
	exit 1
fi

rm -rf errs _obj
exit 0

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
	if go build $file >errs 2>&1; then
		echo 1>&2 misc/cgo/errors/test.bash: BUG: expected cgo to fail but it succeeded
		exit 1
	fi
	if ! test -s errs; then
		echo 1>&2 misc/cgo/errors/test.bash: BUG: expected error output but saw none
		exit 1
	fi
	if ! fgrep $file:$line: errs >/dev/null 2>&1; then
		echo 1>&2 misc/cgo/errors/test.bash: BUG: expected error on line $line but saw:
		cat 1>&2 errs
		exit 1
	fi
}

check err1.go
check err2.go
check err3.go
check issue7757.go
check issue8442.go

rm -rf errs _obj
exit 0

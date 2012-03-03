#!/bin/bash
# Copyright 2012 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e
go build -o testgo

ok=true

# Test that error messages have file:line information
# at beginning of line.
for i in testdata/errmsg/*.go
do
	# TODO: |cat should not be necessary here but is.
	./testgo test $i 2>&1 | cat >err.out || true
	if ! grep -q "^$i:" err.out; then
		echo "$i: missing file:line in error message"
		cat err.out
		ok=false
	fi
done

# Test local (./) imports.
./testgo build -o hello testdata/local/easy.go
./hello >hello.out
if ! grep -q '^easysub\.Hello' hello.out; then
	echo "testdata/local/easy.go did not generate expected output"
	cat hello.out
	ok=false
fi

./testgo build -o hello testdata/local/easysub/main.go
./hello >hello.out
if ! grep -q '^easysub\.Hello' hello.out; then
	echo "testdata/local/easysub/main.go did not generate expected output"
	cat hello.out
	ok=false
fi

./testgo build -o hello testdata/local/hard.go
./hello >hello.out
if ! grep -q '^sub\.Hello' hello.out || ! grep -q '^subsub\.Hello' hello.out ; then
	echo "testdata/local/hard.go did not generate expected output"
	cat hello.out
	ok=false
fi

rm -f err.out hello.out hello

# Test that go install x.go fails.
if ./testgo install testdata/local/easy.go >/dev/null 2>&1; then
	echo "go install testdata/local/easy.go succeeded"
	ok=false
fi

# Test tests with relative imports.
if ! ./testgo test ./testdata/testimport; then
	echo "go test ./testdata/testimport failed"
	ok=false
fi

# Test tests with relative imports in packages synthesized
# from Go files named on the command line.
if ! ./testgo test ./testdata/testimport/*.go; then
	echo "go test ./testdata/testimport/*.go failed"
	ok=false
fi

if $ok; then
	echo PASS
else
	echo FAIL
	exit 1
fi

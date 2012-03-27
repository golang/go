#!/bin/bash
# Copyright 2012 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e
go build -o testgo

ok=true

unset GOPATH
unset GOBIN

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
testlocal() {
	local="$1"
	./testgo build -o hello "testdata/$local/easy.go"
	./hello >hello.out
	if ! grep -q '^easysub\.Hello' hello.out; then
		echo "testdata/$local/easy.go did not generate expected output"
		cat hello.out
		ok=false
	fi
	
	./testgo build -o hello "testdata/$local/easysub/main.go"
	./hello >hello.out
	if ! grep -q '^easysub\.Hello' hello.out; then
		echo "testdata/$local/easysub/main.go did not generate expected output"
		cat hello.out
		ok=false
	fi
	
	./testgo build -o hello "testdata/$local/hard.go"
	./hello >hello.out
	if ! grep -q '^sub\.Hello' hello.out || ! grep -q '^subsub\.Hello' hello.out ; then
		echo "testdata/$local/hard.go did not generate expected output"
		cat hello.out
		ok=false
	fi
	
	rm -f err.out hello.out hello
	
	# Test that go install x.go fails.
	if ./testgo install "testdata/$local/easy.go" >/dev/null 2>&1; then
		echo "go install testdata/$local/easy.go succeeded"
		ok=false
	fi
}

# Test local imports
testlocal local

# Test local imports again, with bad characters in the directory name.
bad='#$%:, &()*;<=>?\^{}'
rm -rf "testdata/$bad"
cp -R testdata/local "testdata/$bad"
testlocal "$bad"
rm -rf "testdata/$bad"

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

# Test that without $GOBIN set, binaries get installed
# into the GOPATH bin directory.
rm -rf testdata/bin
if ! GOPATH=$(pwd)/testdata ./testgo install go-cmd-test; then
	echo "go install go-cmd-test failed"
	ok=false
elif ! test -x testdata/bin/go-cmd-test; then
	echo "go install go-cmd-test did not write to testdata/bin/go-cmd-test"
	ok=false
fi

# And with $GOBIN set, binaries get installed to $GOBIN.
if ! GOBIN=$(pwd)/testdata/bin1 GOPATH=$(pwd)/testdata ./testgo install go-cmd-test; then
	echo "go install go-cmd-test failed"
	ok=false
elif ! test -x testdata/bin1/go-cmd-test; then
	echo "go install go-cmd-test did not write to testdata/bin1/go-cmd-test"
	ok=false
fi

# Without $GOBIN set, installing a program outside $GOPATH should fail
# (there is nowhere to install it).
if ./testgo install testdata/src/go-cmd-test/helloworld.go; then
	echo "go install testdata/src/go-cmd-test/helloworld.go should have failed, did not"
	ok=false
fi

# With $GOBIN set, should install there.
if ! GOBIN=$(pwd)/testdata/bin1 ./testgo install testdata/src/go-cmd-test/helloworld.go; then
	echo "go install testdata/src/go-cmd-test/helloworld.go failed"
	ok=false
elif ! test -x testdata/bin1/helloworld; then
	echo "go install testdata/src/go-cmd-test/helloworld.go did not write testdata/bin1/helloworld"
	ok=false
fi

if $ok; then
	echo PASS
else
	echo FAIL
	exit 1
fi

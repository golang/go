#!/bin/bash
# Copyright 2012 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e
go build -o testgo

ok=true

unset GOPATH
unset GOBIN

# Test that error messages have file:line information at beginning of
# the line. Also test issue 4917: that the error is on stderr.
d=$(mktemp -d -t testgoXXX)
fn=$d/err.go
echo "package main" > $fn
echo 'import "bar"' >> $fn
./testgo run $fn 2>$d/err.out || true
if ! grep -q "^$fn:" $d/err.out; then
	echo "missing file:line in error message"
	cat $d/err.out
	ok=false
fi
rm -r $d

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
	
	rm -f hello.out hello
	
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

# Test installation with relative imports.
if ! ./testgo test -i ./testdata/testimport; then
    echo "go test -i ./testdata/testimport failed"
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

# Reject relative paths in GOPATH.
if GOPATH=. ./testgo build testdata/src/go-cmd-test/helloworld.go; then
    echo 'GOPATH="." go build should have failed, did not'
    ok=false
fi

if GOPATH=:$(pwd)/testdata:. ./testgo build go-cmd-test; then
    echo 'GOPATH=":$(pwd)/testdata:." go build should have failed, did not'
    ok=false
fi

# issue 4104
if [ $(./testgo test fmt fmt fmt fmt fmt | wc -l) -ne 1 ] ; then
    echo 'go test fmt fmt fmt fmt fmt tested the same package multiple times'
    ok=false
fi

# ensure that output of 'go list' is consistent between runs
./testgo list std > test_std.list
if ! ./testgo list std | cmp -s test_std.list - ; then
	echo "go list std ordering is inconsistent"
	ok=false
fi
rm -f test_std.list

# issue 4096. Validate the output of unsucessful go install foo/quxx 
if [ $(./testgo install 'foo/quxx' 2>&1 | grep -c 'cannot find package "foo/quxx" in any of') -ne 1 ] ; then
	echo 'go install foo/quxx expected error: .*cannot find package "foo/quxx" in any of'
	ok=false
fi 
# test GOROOT search failure is reported
if [ $(./testgo install 'foo/quxx' 2>&1 | egrep -c 'foo/quxx \(from \$GOROOT\)$') -ne 1 ] ; then
        echo 'go install foo/quxx expected error: .*foo/quxx (from $GOROOT)'
        ok=false
fi
# test multiple GOPATH entries are reported separately
if [ $(GOPATH=$(pwd)/testdata/a:$(pwd)/testdata/b ./testgo install 'foo/quxx' 2>&1 | egrep -c 'testdata/./src/foo/quxx') -ne 2 ] ; then
        echo 'go install foo/quxx expected error: .*testdata/a/src/foo/quxx (from $GOPATH)\n.*testdata/b/src/foo/quxx'
        ok=false
fi
# test (from $GOPATH) annotation is reported for the first GOPATH entry
if [ $(GOPATH=$(pwd)/testdata/a:$(pwd)/testdata/b ./testgo install 'foo/quxx' 2>&1 | egrep -c 'testdata/a/src/foo/quxx \(from \$GOPATH\)$') -ne 1 ] ; then
        echo 'go install foo/quxx expected error: .*testdata/a/src/foo/quxx (from $GOPATH)'
        ok=false
fi
# but not on the second
if [ $(GOPATH=$(pwd)/testdata/a:$(pwd)/testdata/b ./testgo install 'foo/quxx' 2>&1 | egrep -c 'testdata/b/src/foo/quxx$') -ne 1 ] ; then
        echo 'go install foo/quxx expected error: .*testdata/b/src/foo/quxx'
        ok=false
fi
# test missing GOPATH is reported
if [ $(GOPATH= ./testgo install 'foo/quxx' 2>&1 | egrep -c '\(\$GOPATH not set\)$') -ne 1 ] ; then
        echo 'go install foo/quxx expected error: ($GOPATH not set)'
        ok=false
fi

# issue 4186. go get cannot be used to download packages to $GOROOT
# Test that without GOPATH set, go get should fail
d=$(mktemp -d -t testgoXXX)
mkdir -p $d/src/pkg
if GOPATH= GOROOT=$d ./testgo get -d code.google.com/p/go.codereview/cmd/hgpatch ; then 
	echo 'go get code.google.com/p/go.codereview/cmd/hgpatch should not succeed with $GOPATH unset'
	ok=false
fi	
rm -rf $d
# Test that with GOPATH=$GOROOT, go get should fail
d=$(mktemp -d -t testgoXXX)
mkdir -p $d/src/pkg
if GOPATH=$d GOROOT=$d ./testgo get -d code.google.com/p/go.codereview/cmd/hgpatch ; then
        echo 'go get code.google.com/p/go.codereview/cmd/hgpatch should not succeed with GOPATH=$GOROOT'
        ok=false
fi
rm -rf $d

# issue 3941: args with spaces
d=$(mktemp -d -t testgoXXX)
cat >$d/main.go<<EOF
package main
var extern string
func main() {
	println(extern)
}
EOF
./testgo run -ldflags '-X main.extern "hello world"' $d/main.go 2>hello.out
if ! grep -q '^hello world' hello.out; then
	echo "ldflags -X main.extern 'hello world' failed. Output:"
	cat hello.out
	ok=false
fi
rm -rf $d

# test that go test -cpuprofile leaves binary behind
./testgo test -cpuprofile strings.prof strings || ok=false
if [ ! -x strings.test ]; then
	echo "go test -cpuprofile did not create strings.test"
	ok=false
fi
rm -f strings.prof strings.test

# issue 4568. test that symlinks don't screw things up too badly.
old=$(pwd)
d=$(mktemp -d -t testgoXXX)
mkdir -p $d/src
(
	ln -s $d $d/src/dir1
	cd $d/src/dir1
	echo package p >p.go
	export GOPATH=$d
	if [ "$($old/testgo list -f '{{.Root}}' .)" != "$d" ]; then
		echo got lost in symlink tree:
		pwd
		env|grep WD
		$old/testgo list -json . dir1
		touch $d/failed
	fi		
)
if [ -f $d/failed ]; then
	ok=false
fi
rm -rf $d

# issue 4515.
d=$(mktemp -d -t testgoXXX)
mkdir -p $d/src/example/a $d/src/example/b $d/bin
cat >$d/src/example/a/main.go <<EOF
package main
func main() {}
EOF
cat >$d/src/example/b/main.go <<EOF
// +build mytag

package main
func main() {}
EOF
GOPATH=$d ./testgo install -tags mytag example/a example/b || ok=false
if [ ! -x $d/bin/a -o ! -x $d/bin/b ]; then
	echo go install example/a example/b did not install binaries
	ok=false
fi
rm -f $d/bin/*
GOPATH=$d ./testgo install -tags mytag example/... || ok=false
if [ ! -x $d/bin/a -o ! -x $d/bin/b ]; then
	echo go install example/... did not install binaries
	ok=false
fi
rm -f $d/bin/*go
export GOPATH=$d
if [ "$(./testgo list -tags mytag example/b...)" != "example/b" ]; then
	echo go list example/b did not find example/b
	ok=false
fi
unset GOPATH
rm -rf $d

# issue 4773. case-insensitive collisions
d=$(mktemp -d -t testgoXXX)
export GOPATH=$d
mkdir -p $d/src/example/a $d/src/example/b
cat >$d/src/example/a/a.go <<EOF
package p
import (
	_ "math/rand"
	_ "math/Rand"
)
EOF
if ./testgo list example/a 2>$d/out; then
	echo go list example/a should have failed, did not.
	ok=false
elif ! grep "case-insensitive import collision" $d/out >/dev/null; then
	echo go list example/a did not report import collision.
	ok=false
fi
cat >$d/src/example/b/file.go <<EOF
package b
EOF
cat >$d/src/example/b/FILE.go <<EOF
package b
EOF
if [ $(ls $d/src/example/b | wc -l) = 2 ]; then
	# case-sensitive file system, let directory read find both files
	args="example/b"
else
	# case-insensitive file system, list files explicitly on command line.
	args="$d/src/example/b/file.go $d/src/example/b/FILE.go"
fi
if ./testgo list $args 2>$d/out; then
	echo go list example/b should have failed, did not.
	ok=false
elif ! grep "case-insensitive file name collision" $d/out >/dev/null; then
	echo go list example/b did not report file name collision.
	ok=false
fi
unset GOPATH
rm -rf $d

# Only succeeds if source order is preserved.
./testgo test testdata/example[12]_test.go

# clean up
rm -rf testdata/bin testdata/bin1
rm -f testgo

if $ok; then
	echo PASS
else
	echo FAIL
	exit 1
fi

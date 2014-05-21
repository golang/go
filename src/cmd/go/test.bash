#!/bin/bash
# Copyright 2012 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e
go build -o testgo
go() {
	echo TEST ERROR: ran go, not testgo: go "$@" >&2
	exit 2
}

started=false
TEST() {
	if $started; then
		stop
	fi
	echo TEST: "$@"
	started=true
	ok=true
}
stop() {
	if ! $started; then
		echo TEST ERROR: stop missing start >&2
		exit 2
	fi
	started=false
	if $ok; then
		echo PASS
	else
		echo FAIL
		allok=false
	fi
}

ok=true
allok=true

unset GOBIN
unset GOPATH
unset GOROOT

TEST 'file:line in error messages'
# Test that error messages have file:line information at beginning of
# the line. Also test issue 4917: that the error is on stderr.
d=$(TMPDIR=/var/tmp mktemp -d -t testgoXXX)
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
	TEST local imports $2 '(easy)'
	./testgo build -o hello "testdata/$local/easy.go"
	./hello >hello.out
	if ! grep -q '^easysub\.Hello' hello.out; then
		echo "testdata/$local/easy.go did not generate expected output"
		cat hello.out
		ok=false
	fi
	
	TEST local imports $2 '(easysub)'
	./testgo build -o hello "testdata/$local/easysub/main.go"
	./hello >hello.out
	if ! grep -q '^easysub\.Hello' hello.out; then
		echo "testdata/$local/easysub/main.go did not generate expected output"
		cat hello.out
		ok=false
	fi
	
	TEST local imports $2 '(hard)'
	./testgo build -o hello "testdata/$local/hard.go"
	./hello >hello.out
	if ! grep -q '^sub\.Hello' hello.out || ! grep -q '^subsub\.Hello' hello.out ; then
		echo "testdata/$local/hard.go did not generate expected output"
		cat hello.out
		ok=false
	fi
	
	rm -f hello.out hello
	
	# Test that go install x.go fails.
	TEST local imports $2 '(go install should fail)'
	if ./testgo install "testdata/$local/easy.go" >/dev/null 2>&1; then
		echo "go install testdata/$local/easy.go succeeded"
		ok=false
	fi
}

# Test local imports
testlocal local ''

# Test local imports again, with bad characters in the directory name.
bad='#$%:, &()*;<=>?\^{}'
rm -rf "testdata/$bad"
cp -R testdata/local "testdata/$bad"
testlocal "$bad" 'with bad characters in path'
rm -rf "testdata/$bad"

TEST error message for syntax error in test go file says FAIL
export GOPATH=$(pwd)/testdata
if ./testgo test syntaxerror 2>testdata/err; then
	echo 'go test syntaxerror succeeded'
	ok=false
elif ! grep FAIL testdata/err >/dev/null; then
	echo 'go test did not say FAIL:'
	cat testdata/err
	ok=false
fi
rm -f ./testdata/err
unset GOPATH

TEST wildcards do not look in useless directories
export GOPATH=$(pwd)/testdata
if ./testgo list ... >testdata/err 2>&1; then
	echo "go list ... succeeded"
	ok=false
elif ! grep badpkg testdata/err >/dev/null; then
	echo "go list ... failure does not mention badpkg"
	cat testdata/err
	ok=false
elif ! ./testgo list m... >testdata/err 2>&1; then
	echo "go list m... failed"
	ok=false
fi
rm -rf ./testdata/err
unset GOPATH

# Test tests with relative imports.
TEST relative imports '(go test)'
if ! ./testgo test ./testdata/testimport; then
	echo "go test ./testdata/testimport failed"
	ok=false
fi

# Test installation with relative imports.
TEST relative imports '(go test -i)'
if ! ./testgo test -i ./testdata/testimport; then
    echo "go test -i ./testdata/testimport failed"
    ok=false
fi

# Test tests with relative imports in packages synthesized
# from Go files named on the command line.
TEST relative imports in command-line package
if ! ./testgo test ./testdata/testimport/*.go; then
	echo "go test ./testdata/testimport/*.go failed"
	ok=false
fi

TEST version control error message includes correct directory
export GOPATH=$(pwd)/testdata/shadow/root1
if ./testgo get -u foo 2>testdata/err; then
	echo "go get -u foo succeeded unexpectedly"
	ok=false
elif ! grep testdata/shadow/root1/src/foo testdata/err >/dev/null; then
	echo "go get -u error does not mention shadow/root1/src/foo:"
	cat testdata/err
	ok=false
fi
unset GOPATH

TEST go install fails with no buildable files
export GOPATH=$(pwd)/testdata
export CGO_ENABLED=0
if ./testgo install cgotest 2>testdata/err; then
	echo "go install cgotest succeeded unexpectedly"
elif ! grep 'no buildable Go source files' testdata/err >/dev/null; then
	echo "go install cgotest did not report 'no buildable Go source files'"
	cat testdata/err
	ok=false
fi
unset CGO_ENABLED
unset GOPATH

# Test that without $GOBIN set, binaries get installed
# into the GOPATH bin directory.
TEST install into GOPATH
rm -rf testdata/bin
if ! GOPATH=$(pwd)/testdata ./testgo install go-cmd-test; then
	echo "go install go-cmd-test failed"
	ok=false
elif ! test -x testdata/bin/go-cmd-test; then
	echo "go install go-cmd-test did not write to testdata/bin/go-cmd-test"
	ok=false
fi

TEST package main_test imports archive not binary
export GOBIN=$(pwd)/testdata/bin
mkdir -p $GOBIN
export GOPATH=$(pwd)/testdata
touch ./testdata/src/main_test/m.go
if ! ./testgo test main_test; then
	echo "go test main_test failed without install"
	ok=false
elif ! ./testgo install main_test; then
	echo "go test main_test failed"
	ok=false
elif [ "$(./testgo list -f '{{.Stale}}' main_test)" != false ]; then
	echo "after go install, main listed as stale"
	ok=false
elif ! ./testgo test main_test; then
	echo "go test main_test failed after install"
	ok=false
fi
rm -rf $GOBIN
unset GOBIN

# And with $GOBIN set, binaries get installed to $GOBIN.
TEST install into GOBIN
if ! GOBIN=$(pwd)/testdata/bin1 GOPATH=$(pwd)/testdata ./testgo install go-cmd-test; then
	echo "go install go-cmd-test failed"
	ok=false
elif ! test -x testdata/bin1/go-cmd-test; then
	echo "go install go-cmd-test did not write to testdata/bin1/go-cmd-test"
	ok=false
fi

# Without $GOBIN set, installing a program outside $GOPATH should fail
# (there is nowhere to install it).
TEST install without destination fails
if ./testgo install testdata/src/go-cmd-test/helloworld.go 2>testdata/err; then
	echo "go install testdata/src/go-cmd-test/helloworld.go should have failed, did not"
	ok=false
elif ! grep 'no install location for .go files listed on command line' testdata/err; then
	echo "wrong error:"
	cat testdata/err
	ok=false
fi
rm -f testdata/err

# With $GOBIN set, should install there.
TEST install to GOBIN '(command-line package)'
if ! GOBIN=$(pwd)/testdata/bin1 ./testgo install testdata/src/go-cmd-test/helloworld.go; then
	echo "go install testdata/src/go-cmd-test/helloworld.go failed"
	ok=false
elif ! test -x testdata/bin1/helloworld; then
	echo "go install testdata/src/go-cmd-test/helloworld.go did not write testdata/bin1/helloworld"
	ok=false
fi

TEST godoc installs into GOBIN
d=$(mktemp -d -t testgoXXX)
export GOPATH=$d
mkdir $d/gobin
GOBIN=$d/gobin ./testgo get code.google.com/p/go.tools/cmd/godoc
if [ ! -x $d/gobin/godoc ]; then
	echo did not install godoc to '$GOBIN'
	GOBIN=$d/gobin ./testgo list -f 'Target: {{.Target}}' code.google.com/p/go.tools/cmd/godoc
	ok=false
fi

TEST godoc installs into GOROOT
GOROOT=$(./testgo env GOROOT)
rm -f $GOROOT/bin/godoc
./testgo install code.google.com/p/go.tools/cmd/godoc
if [ ! -x $GOROOT/bin/godoc ]; then
	echo did not install godoc to '$GOROOT/bin'
	./testgo list -f 'Target: {{.Target}}' code.google.com/p/go.tools/cmd/godoc
	ok=false
fi

TEST cmd/fix installs into tool
GOOS=$(./testgo env GOOS)
GOARCH=$(./testgo env GOARCH)
rm -f $GOROOT/pkg/tool/${GOOS}_${GOARCH}/fix
./testgo install cmd/fix
if [ ! -x $GOROOT/pkg/tool/${GOOS}_${GOARCH}/fix ]; then
	echo 'did not install cmd/fix to $GOROOT/pkg/tool'
	GOBIN=$d/gobin ./testgo list -f 'Target: {{.Target}}' cmd/fix
	ok=false
fi
rm -f $GOROOT/pkg/tool/${GOOS}_${GOARCH}/fix
GOBIN=$d/gobin ./testgo install cmd/fix
if [ ! -x $GOROOT/pkg/tool/${GOOS}_${GOARCH}/fix ]; then
	echo 'did not install cmd/fix to $GOROOT/pkg/tool with $GOBIN set'
	GOBIN=$d/gobin ./testgo list -f 'Target: {{.Target}}' cmd/fix
	ok=false
fi

TEST gopath program installs into GOBIN
mkdir $d/src/progname
echo 'package main; func main() {}' >$d/src/progname/p.go
GOBIN=$d/gobin ./testgo install progname
if [ ! -x $d/gobin/progname ]; then
	echo 'did not install progname to $GOBIN/progname'
	./testgo list -f 'Target: {{.Target}}' cmd/api
	ok=false
fi
rm -f $d/gobin/progname $d/bin/progname

TEST gopath program installs into GOPATH/bin
./testgo install progname
if [ ! -x $d/bin/progname ]; then
	echo 'did not install progname to $GOPATH/bin/progname'
	./testgo list -f 'Target: {{.Target}}' progname
	ok=false
fi

unset GOPATH
rm -rf $d

# Reject relative paths in GOPATH.
TEST reject relative paths in GOPATH '(command-line package)'
if GOPATH=. ./testgo build testdata/src/go-cmd-test/helloworld.go; then
    echo 'GOPATH="." go build should have failed, did not'
    ok=false
fi

TEST reject relative paths in GOPATH 
if GOPATH=:$(pwd)/testdata:. ./testgo build go-cmd-test; then
    echo 'GOPATH=":$(pwd)/testdata:." go build should have failed, did not'
    ok=false
fi

# issue 4104
TEST go test with package listed multiple times
if [ $(./testgo test fmt fmt fmt fmt fmt | wc -l) -ne 1 ] ; then
    echo 'go test fmt fmt fmt fmt fmt tested the same package multiple times'
    ok=false
fi

# ensure that output of 'go list' is consistent between runs
TEST go list is consistent
./testgo list std > test_std.list
if ! ./testgo list std | cmp -s test_std.list - ; then
	echo "go list std ordering is inconsistent"
	ok=false
fi
rm -f test_std.list

# issue 4096. Validate the output of unsuccessful go install foo/quxx 
TEST unsuccessful go install should mention missing package
if [ $(./testgo install 'foo/quxx' 2>&1 | grep -c 'cannot find package "foo/quxx" in any of') -ne 1 ] ; then
	echo 'go install foo/quxx expected error: .*cannot find package "foo/quxx" in any of'
	ok=false
fi 
# test GOROOT search failure is reported
TEST GOROOT search failure reporting
if [ $(./testgo install 'foo/quxx' 2>&1 | egrep -c 'foo/quxx \(from \$GOROOT\)$') -ne 1 ] ; then
        echo 'go install foo/quxx expected error: .*foo/quxx (from $GOROOT)'
        ok=false
fi
# test multiple GOPATH entries are reported separately
TEST multiple GOPATH entries reported separately
if [ $(GOPATH=$(pwd)/testdata/a:$(pwd)/testdata/b ./testgo install 'foo/quxx' 2>&1 | egrep -c 'testdata/./src/foo/quxx') -ne 2 ] ; then
        echo 'go install foo/quxx expected error: .*testdata/a/src/foo/quxx (from $GOPATH)\n.*testdata/b/src/foo/quxx'
        ok=false
fi
# test (from $GOPATH) annotation is reported for the first GOPATH entry
TEST mention GOPATH in first GOPATH entry
if [ $(GOPATH=$(pwd)/testdata/a:$(pwd)/testdata/b ./testgo install 'foo/quxx' 2>&1 | egrep -c 'testdata/a/src/foo/quxx \(from \$GOPATH\)$') -ne 1 ] ; then
        echo 'go install foo/quxx expected error: .*testdata/a/src/foo/quxx (from $GOPATH)'
        ok=false
fi
# but not on the second
TEST but not the second entry
if [ $(GOPATH=$(pwd)/testdata/a:$(pwd)/testdata/b ./testgo install 'foo/quxx' 2>&1 | egrep -c 'testdata/b/src/foo/quxx$') -ne 1 ] ; then
        echo 'go install foo/quxx expected error: .*testdata/b/src/foo/quxx'
        ok=false
fi
# test missing GOPATH is reported
TEST missing GOPATH is reported
if [ $(GOPATH= ./testgo install 'foo/quxx' 2>&1 | egrep -c '\(\$GOPATH not set\)$') -ne 1 ] ; then
        echo 'go install foo/quxx expected error: ($GOPATH not set)'
        ok=false
fi

# issue 4186. go get cannot be used to download packages to $GOROOT
# Test that without GOPATH set, go get should fail
TEST without GOPATH, go get fails
d=$(mktemp -d -t testgoXXX)
mkdir -p $d/src/pkg
if GOPATH= GOROOT=$d ./testgo get -d code.google.com/p/go.codereview/cmd/hgpatch ; then 
	echo 'go get code.google.com/p/go.codereview/cmd/hgpatch should not succeed with $GOPATH unset'
	ok=false
fi	
rm -rf $d

# Test that with GOPATH=$GOROOT, go get should fail
TEST with GOPATH=GOROOT, go get fails
d=$(mktemp -d -t testgoXXX)
mkdir -p $d/src/pkg
if GOPATH=$d GOROOT=$d ./testgo get -d code.google.com/p/go.codereview/cmd/hgpatch ; then
        echo 'go get code.google.com/p/go.codereview/cmd/hgpatch should not succeed with GOPATH=$GOROOT'
        ok=false
fi
rm -rf $d

TEST ldflags arguments with spaces '(issue 3941)'
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
rm -rf $d hello.out

TEST go test -cpuprofile leaves binary behind
./testgo test -cpuprofile strings.prof strings || ok=false
if [ ! -x strings.test ]; then
	echo "go test -cpuprofile did not create strings.test"
	ok=false
fi
rm -f strings.prof strings.test

TEST symlinks do not confuse go list '(issue 4568)'
old=$(pwd)
tmp=$(cd /tmp && pwd -P)
d=$(TMPDIR=$tmp mktemp -d -t testgoXXX)
mkdir -p $d/src
(
	ln -s $d $d/src/dir1
	cd $d/src
	echo package p >dir1/p.go
	export GOPATH=$d
	if [ "$($old/testgo list -f '{{.Root}}' dir1)" != "$d" ]; then
		echo Confused by symlinks.
		echo "Package in current directory $(pwd) should have Root $d"
		env|grep WD
		$old/testgo list -json . dir1
		touch $d/failed
	fi		
)
if [ -f $d/failed ]; then
	ok=false
fi
rm -rf $d

TEST 'install with tags (issue 4515)'
d=$(TMPDIR=/var/tmp mktemp -d -t testgoXXX)
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

TEST case collisions '(issue 4773)'
d=$(TMPDIR=/var/tmp mktemp -d -t testgoXXX)
export GOPATH=$d
mkdir -p $d/src/example/{a/pkg,a/Pkg,b}
cat >$d/src/example/a/a.go <<EOF
package p
import (
	_ "example/a/pkg"
	_ "example/a/Pkg"
)
EOF
cat >$d/src/example/a/pkg/pkg.go <<EOF
package pkg
EOF
cat >$d/src/example/a/Pkg/pkg.go <<EOF
package pkg
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

TEST go get cover
./testgo get code.google.com/p/go.tools/cmd/cover || ok=false

unset GOPATH
rm -rf $d

TEST shadowing logic
export GOPATH=$(pwd)/testdata/shadow/root1:$(pwd)/testdata/shadow/root2

# The math in root1 is not "math" because the standard math is.
cdir=$(./testgo list -f '({{.ImportPath}}) ({{.ConflictDir}})' ./testdata/shadow/root1/src/math)
if [ "$cdir" != "(_$(pwd)/testdata/shadow/root1/src/math) ($GOROOT/src/pkg/math)" ]; then
	echo shadowed math is not shadowed: "$cdir"
	ok=false
fi

# The foo in root1 is "foo".
cdir=$(./testgo list -f '({{.ImportPath}}) ({{.ConflictDir}})' ./testdata/shadow/root1/src/foo)
if [ "$cdir" != "(foo) ()" ]; then
	echo unshadowed foo is shadowed: "$cdir"
	ok=false
fi

# The foo in root2 is not "foo" because the foo in root1 got there first.
cdir=$(./testgo list -f '({{.ImportPath}}) ({{.ConflictDir}})' ./testdata/shadow/root2/src/foo)
if [ "$cdir" != "(_$(pwd)/testdata/shadow/root2/src/foo) ($(pwd)/testdata/shadow/root1/src/foo)" ]; then
	echo shadowed foo is not shadowed: "$cdir"
	ok=false
fi

# The error for go install should mention the conflicting directory.
err=$(! ./testgo install ./testdata/shadow/root2/src/foo 2>&1)
if [ "$err" != "go install: no install location for $(pwd)/testdata/shadow/root2/src/foo: hidden by $(pwd)/testdata/shadow/root1/src/foo" ]; then
	echo wrong shadowed install error: "$err"
	ok=false
fi

# Only succeeds if source order is preserved.
TEST source file name order preserved
./testgo test testdata/example[12]_test.go || ok=false

# Check that coverage analysis works at all.
# Don't worry about the exact numbers but require not 0.0%.
checkcoverage() {
	if grep '[^0-9]0\.0%' testdata/cover.txt >/dev/null; then
		echo 'some coverage results are 0.0%'
		ok=false
	fi
	cat testdata/cover.txt
	rm -f testdata/cover.txt
}
	
TEST coverage runs
./testgo test -short -coverpkg=strings strings regexp >testdata/cover.txt 2>&1 || ok=false
./testgo test -short -cover strings math regexp >>testdata/cover.txt 2>&1 || ok=false
checkcoverage

# Check that coverage analysis uses set mode.
TEST coverage uses set mode
if ./testgo test -short -cover encoding/binary -coverprofile=testdata/cover.out >testdata/cover.txt 2>&1; then
	if ! grep -q 'mode: set' testdata/cover.out; then
		ok=false
	fi
	checkcoverage
else
	ok=false
fi
rm -f testdata/cover.out testdata/cover.txt

TEST coverage uses atomic mode for -race.
if ./testgo test -short -race -cover encoding/binary -coverprofile=testdata/cover.out >testdata/cover.txt 2>&1; then
	if ! grep -q 'mode: atomic' testdata/cover.out; then
		ok=false
	fi
	checkcoverage
else
	ok=false
fi
rm -f testdata/cover.out

TEST coverage uses actual setting to override even for -race.
if ./testgo test -short -race -cover encoding/binary -covermode=count -coverprofile=testdata/cover.out >testdata/cover.txt 2>&1; then
	if ! grep -q 'mode: count' testdata/cover.out; then
		ok=false
	fi
	checkcoverage
else
	ok=false
fi
rm -f testdata/cover.out

TEST coverage with cgo
d=$(TMPDIR=/var/tmp mktemp -d -t testgoXXX)
./testgo test -short -cover ./testdata/cgocover >testdata/cover.txt 2>&1 || ok=false
checkcoverage

TEST cgo depends on syscall
rm -rf $GOROOT/pkg/*_race
d=$(TMPDIR=/var/tmp mktemp -d -t testgoXXX)
export GOPATH=$d
mkdir -p $d/src/foo
echo '
package foo
//#include <stdio.h>
import "C"
' >$d/src/foo/foo.go
./testgo build -race foo || ok=false
rm -rf $d
unset GOPATH

TEST cgo shows full path names
d=$(TMPDIR=/var/tmp mktemp -d -t testgoXXX)
export GOPATH=$d
mkdir -p $d/src/x/y/dirname
echo '
package foo
import "C"
func f() {
' >$d/src/x/y/dirname/foo.go
if ./testgo build x/y/dirname >$d/err 2>&1; then
	echo build succeeded unexpectedly.
	ok=false
elif ! grep x/y/dirname $d/err >/dev/null; then
	echo error did not use full path.
	cat $d/err
	ok=false
fi
rm -rf $d
unset GOPATH

TEST 'cgo handles -Wl,$ORIGIN'
d=$(TMPDIR=/var/tmp mktemp -d -t testgoXXX)
export GOPATH=$d
mkdir -p $d/src/origin
echo '
package origin
// #cgo !darwin LDFLAGS: -Wl,-rpath -Wl,$ORIGIN
// void f(void) {}
import "C"

func f() { C.f() }
' >$d/src/origin/origin.go
if ! ./testgo build origin; then
	echo build failed
	ok=false
fi
rm -rf $d
unset GOPATH

TEST 'Issue 6480: "go test -c -test.bench=XXX fmt" should not hang'
if ! ./testgo test -c -test.bench=XXX fmt; then
	echo build test failed
	ok=false
fi
rm -f fmt.test

TEST 'Issue 7573: cmd/cgo: undefined reference when linking a C-library using gccgo'
d=$(mktemp -d -t testgoXXX)
export GOPATH=$d
mkdir -p $d/src/cgoref
ldflags="-L alibpath -lalib"
echo "
package main
// #cgo LDFLAGS: $ldflags
// void f(void) {}
import \"C\"

func main() { C.f() }
" >$d/src/cgoref/cgoref.go
go_cmds="$(./testgo build -n -compiler gccgo cgoref 2>&1 1>/dev/null)"
ldflags_count="$(echo "$go_cmds" | egrep -c "^gccgo.*$(echo $ldflags | sed -e 's/-/\\-/g')" || true)"
if [ "$ldflags_count" -lt 1 ]; then
	echo "No Go-inline "#cgo LDFLAGS:" (\"$ldflags\") passed to gccgo linking stage."
	ok=false
fi
rm -rf $d
unset ldflags_count
unset go_cmds
unset ldflags
unset GOPATH

TEST list template can use context function
if ! ./testgo list -f "GOARCH: {{context.GOARCH}}"; then 
	echo unable to use context in list template
	ok=false
fi

TEST 'Issue 7108: cmd/go: "go test" should fail if package does not build'
export GOPATH=$(pwd)/testdata
if ./testgo test notest >/dev/null 2>&1; then
	echo 'go test notest succeeded, but should fail'
	ok=false
fi
unset GOPATH

TEST 'Issue 6844: cmd/go: go test -a foo does not rebuild regexp'
if ! ./testgo test -x -a -c testdata/dep_test.go 2>deplist; then
	echo "go test -x -a -c testdata/dep_test.go failed"
	ok=false
elif ! grep -q regexp deplist; then
	echo "go test -x -a -c testdata/dep_test.go did not rebuild regexp"
	ok=false
fi
rm -f deplist
rm -f deps.test

TEST list template can use context function
if ! ./testgo list -f "GOARCH: {{context.GOARCH}}"; then 
	echo unable to use context in list template
	ok=false
fi

TEST build -i installs dependencies
d=$(TMPDIR=/var/tmp mktemp -d -t testgoXXX)
export GOPATH=$d
mkdir -p $d/src/x/y/foo $d/src/x/y/bar
echo '
package foo
func F() {}
' >$d/src/x/y/foo/foo.go
echo '
package bar
import "x/y/foo"
func F() { foo.F() }
' >$d/src/x/y/bar/bar.go
if ! ./testgo build -v -i x/y/bar &> $d/err; then
	echo build -i failed
	cat $d/err
	ok=false
elif ! grep x/y/foo $d/err >/dev/null; then
	echo first build -i did not build x/y/foo
	cat $d/err
	ok=false
fi
if ! ./testgo build -v -i x/y/bar &> $d/err; then
	echo second build -i failed
	cat $d/err
	ok=false
elif grep x/y/foo $d/err >/dev/null; then
	echo second build -i built x/y/foo
	cat $d/err
	ok=false
fi
rm -rf $d
unset GOPATH

TEST 'go build in test-only directory fails with a good error'
if ./testgo build ./testdata/testonly 2>testdata/err.out; then
	echo "go build ./testdata/testonly succeeded, should have failed"
	ok=false
elif ! grep 'no buildable Go' testdata/err.out >/dev/null; then
	echo "go build ./testdata/testonly produced unexpected error:"
	cat testdata/err.out
	ok=false
fi
rm -f testdata/err.out

TEST 'go test detects test-only import cycles'
export GOPATH=$(pwd)/testdata
if ./testgo test -c testcycle/p3 2>testdata/err.out; then
	echo "go test testcycle/p3 succeeded, should have failed"
	ok=false
elif ! grep 'import cycle not allowed in test' testdata/err.out >/dev/null; then
	echo "go test testcycle/p3 produced unexpected error:"
	cat testdata/err.out
	ok=false
fi
rm -f testdata/err.out
unset GOPATH

TEST 'go test foo_test.go works'
if ! ./testgo test testdata/standalone_test.go; then
	echo "go test testdata/standalone_test.go failed"
	ok=false
fi

TEST 'go test xtestonly works'
export GOPATH=$(pwd)/testdata
./testgo clean -i xtestonly
if ! ./testgo test xtestonly >/dev/null; then
	echo "go test xtestonly failed"
	ok=false
fi
unset GOPATH


# clean up
if $started; then stop; fi
rm -rf testdata/bin testdata/bin1
rm -f testgo

if $allok; then
	echo PASS
else
	echo FAIL
	exit 1
fi

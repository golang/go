#!/bin/bash
# Copyright 2012 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e
go build -tags testgo -o testgo
go() {
	echo TEST ERROR: ran go, not testgo: go "$@" >&2
	exit 2
}

started=false
testdesc=""
nl="
"
TEST() {
	if $started; then
		stop
	fi
	echo TEST: "$@"
	testdesc="$@"
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
		testfail="$testfail	$testdesc$nl"
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

TEST 'program name in crash messages'
linker=$(./testgo env GOCHAR)l
d=$(TMPDIR=/var/tmp mktemp -d -t testgoXXX)
./testgo build -ldflags -crash_for_testing $(./testgo env GOROOT)/test/helloworld.go 2>$d/err.out || true
if ! grep -q "/tool/.*/$linker" $d/err.out; then
	echo "missing linker name in error message"
	cat $d/err.out
	ok=false
fi
rm -r $d

TEST broken tests without Test functions all fail
d=$(mktemp -d -t testgoXXX)
./testgo test ./testdata/src/badtest/... >$d/err 2>&1 || true
if grep -q '^ok' $d/err; then
	echo test passed unexpectedly:
	grep '^ok' $d/err
	ok=false
elif ! grep -q 'FAIL.*badtest/badexec' $d/err || ! grep -q 'FAIL.*badtest/badsyntax' $d/err || ! grep -q 'FAIL.*badtest/badvar' $d/err; then
	echo test did not run everything
	cat $d/err
	ok=false
fi
rm -rf $d

TEST 'go build -a in dev branch'
./testgo install math || ok=false # should be up to date already but just in case
d=$(TMPDIR=/var/tmp mktemp -d -t testgoXXX)
if ! TESTGO_IS_GO_RELEASE=0 ./testgo build -v -a math 2>$d/err.out; then
	cat $d/err.out
	ok=false
elif ! grep -q runtime $d/err.out; then
	echo "testgo build -a math in dev branch DID NOT build runtime, but should have"
	cat $d/err.out
	ok=false
fi
rm -r $d

TEST 'go build -a in release branch'
./testgo install math || ok=false # should be up to date already but just in case
d=$(TMPDIR=/var/tmp mktemp -d -t testgoXXX)
if ! TESTGO_IS_GO_RELEASE=1 ./testgo build -v -a math 2>$d/err.out; then
	cat $d/err.out
	ok=false
elif grep -q runtime $d/err.out; then
	echo "testgo build -a math in dev branch DID build runtime, but should NOT have"
	cat $d/err.out
	ok=false
fi
rm -r $d

# Test local (./) imports.
testlocal() {
	local="$1"
	TEST local imports $2 '(easy)'
	./testgo build -o hello "testdata/$local/easy.go" || ok=false
	./hello >hello.out || ok=false
	if ! grep -q '^easysub\.Hello' hello.out; then
		echo "testdata/$local/easy.go did not generate expected output"
		cat hello.out
		ok=false
	fi
	
	TEST local imports $2 '(easysub)'
	./testgo build -o hello "testdata/$local/easysub/main.go" || ok=false
	./hello >hello.out || ok=false
	if ! grep -q '^easysub\.Hello' hello.out; then
		echo "testdata/$local/easysub/main.go did not generate expected output"
		cat hello.out
		ok=false
	fi
	
	TEST local imports $2 '(hard)'
	./testgo build -o hello "testdata/$local/hard.go" || ok=false
	./hello >hello.out || ok=false
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

TEST 'internal packages in $GOROOT are respected'
if ./testgo build -v ./testdata/testinternal >testdata/std.out 2>&1; then
	echo "go build ./testdata/testinternal succeeded incorrectly"
	ok=false
elif ! grep 'use of internal package not allowed' testdata/std.out >/dev/null; then
	echo "wrong error message for testdata/testinternal"
	cat std.out
	ok=false
fi

TEST 'internal packages outside $GOROOT are not respected'
if ! ./testgo build -v ./testdata/testinternal2; then
	echo "go build ./testdata/testinternal2 failed"
	ok=false
fi

# Test that 'go get -u' reports moved packages.
testmove() {
	vcs=$1
	url=$2
	base=$3
	config=$4

	TEST go get -u notices $vcs package that moved
	d=$(mktemp -d -t testgoXXX)
	mkdir -p $d/src
	if ! GOPATH=$d ./testgo get -d $url; then
		echo 'go get -d $url failed'
		ok=false
	elif ! GOPATH=$d ./testgo get -d -u $url; then
		echo 'go get -d -u $url failed'
		ok=false
	else
		set +e
		case "$vcs" in
		svn)
			# SVN doesn't believe in text files so we can't just edit the config.
			# Check out a different repo into the wrong place.
			rm -rf $d/src/code.google.com/p/rsc-svn
			GOPATH=$d ./testgo get -d -u code.google.com/p/rsc-svn2/trunk
			mv $d/src/code.google.com/p/rsc-svn2 $d/src/code.google.com/p/rsc-svn
			;;
		*)
			echo '1,$s;'"$base"';'"$base"'XXX;
w
q' | ed $d/src/$config >/dev/null 2>&1
		esac
		set -e

		if GOPATH=$d ./testgo get -d -u $url 2>$d/err; then
			echo "go get -d -u $url succeeded with wrong remote repo"
			cat $d/err
			ok=false
		elif ! grep 'should be from' $d/err >/dev/null; then
			echo "go get -d -u $url failed for wrong reason"
			cat $d/err
			ok=false
		fi
		
		if GOPATH=$d ./testgo get -d -f -u $url 2>$d/err; then
			echo "go get -d -u $url succeeded with wrong remote repo"
			cat $d/err
			ok=false
		elif ! egrep -i 'validating server certificate|not found' $d/err >/dev/null; then
			echo "go get -d -f -u $url failed for wrong reason"
			cat $d/err
			ok=false
		fi
	fi
	rm -rf $d
}

testmove hg rsc.io/x86/x86asm x86 rsc.io/x86/.hg/hgrc
testmove git rsc.io/pdf pdf rsc.io/pdf/.git/config
testmove svn code.google.com/p/rsc-svn/trunk - -

export GOPATH=$(pwd)/testdata/importcom
TEST 'import comment - match'
if ! ./testgo build ./testdata/importcom/works.go; then
	echo 'go build ./testdata/importcom/works.go failed'
	ok=false
fi
TEST 'import comment - mismatch'
if ./testgo build ./testdata/importcom/wrongplace.go 2>testdata/err; then
	echo 'go build ./testdata/importcom/wrongplace.go suceeded'
	ok=false
elif ! grep 'wrongplace expects import "my/x"' testdata/err >/dev/null; then
	echo 'go build did not mention incorrect import:'
	cat testdata/err
	ok=false
fi
TEST 'import comment - syntax error'
if ./testgo build ./testdata/importcom/bad.go 2>testdata/err; then
	echo 'go build ./testdata/importcom/bad.go suceeded'
	ok=false
elif ! grep 'cannot parse import comment' testdata/err >/dev/null; then
	echo 'go build did not mention syntax error:'
	cat testdata/err
	ok=false
fi
TEST 'import comment - conflict'
if ./testgo build ./testdata/importcom/conflict.go 2>testdata/err; then
	echo 'go build ./testdata/importcom/conflict.go suceeded'
	ok=false
elif ! grep 'found import comments' testdata/err >/dev/null; then
	echo 'go build did not mention comment conflict:'
	cat testdata/err
	ok=false
fi
rm -f ./testdata/err
unset GOPATH

export GOPATH=$(pwd)/testdata/src
TEST disallowed C source files
export GOPATH=$(pwd)/testdata
if ./testgo build badc 2>testdata/err; then
	echo 'go build badc succeeded'
	ok=false
elif ! grep 'C source files not allowed' testdata/err >/dev/null; then
	echo 'go test did not say C source files not allowed:'
	cat testdata/err
	ok=false
fi
rm -f ./testdata/err
unset GOPATH

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
GOBIN=$d/gobin ./testgo get golang.org/x/tools/cmd/godoc || ok=false
if [ ! -x $d/gobin/godoc ]; then
	echo did not install godoc to '$GOBIN'
	GOBIN=$d/gobin ./testgo list -f 'Target: {{.Target}}' golang.org/x/tools/cmd/godoc || true
	ok=false
fi

TEST godoc installs into GOROOT
GOROOT=$(./testgo env GOROOT)
rm -f $GOROOT/bin/godoc
./testgo install golang.org/x/tools/cmd/godoc || ok=false
if [ ! -x $GOROOT/bin/godoc ]; then
	echo did not install godoc to '$GOROOT/bin'
	./testgo list -f 'Target: {{.Target}}' golang.org/x/tools/cmd/godoc || true
	ok=false
fi

TEST cmd/fix installs into tool
GOOS=$(./testgo env GOOS)
GOARCH=$(./testgo env GOARCH)
rm -f $GOROOT/pkg/tool/${GOOS}_${GOARCH}/fix
./testgo install cmd/fix || ok=false
if [ ! -x $GOROOT/pkg/tool/${GOOS}_${GOARCH}/fix ]; then
	echo 'did not install cmd/fix to $GOROOT/pkg/tool'
	GOBIN=$d/gobin ./testgo list -f 'Target: {{.Target}}' cmd/fix || true
	ok=false
fi
rm -f $GOROOT/pkg/tool/${GOOS}_${GOARCH}/fix
GOBIN=$d/gobin ./testgo install cmd/fix || ok=false
if [ ! -x $GOROOT/pkg/tool/${GOOS}_${GOARCH}/fix ]; then
	echo 'did not install cmd/fix to $GOROOT/pkg/tool with $GOBIN set'
	GOBIN=$d/gobin ./testgo list -f 'Target: {{.Target}}' cmd/fix || true
	ok=false
fi

TEST gopath program installs into GOBIN
mkdir $d/src/progname
echo 'package main; func main() {}' >$d/src/progname/p.go
GOBIN=$d/gobin ./testgo install progname || ok=false
if [ ! -x $d/gobin/progname ]; then
	echo 'did not install progname to $GOBIN/progname'
	./testgo list -f 'Target: {{.Target}}' cmd/api || true
	ok=false
fi
rm -f $d/gobin/progname $d/bin/progname

TEST gopath program installs into GOPATH/bin
./testgo install progname || ok=false
if [ ! -x $d/bin/progname ]; then
	echo 'did not install progname to $GOPATH/bin/progname'
	./testgo list -f 'Target: {{.Target}}' progname || true
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
./testgo list std > test_std.list || ok=false
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
mkdir -p $d/src
if GOPATH= GOROOT=$d ./testgo get -d golang.org/x/codereview/cmd/hgpatch ; then 
	echo 'go get golang.org/x/codereview/cmd/hgpatch should not succeed with $GOPATH unset'
	ok=false
fi	
rm -rf $d

# Test that with GOPATH=$GOROOT, go get should fail
TEST with GOPATH=GOROOT, go get fails
d=$(mktemp -d -t testgoXXX)
mkdir -p $d/src
if GOPATH=$d GOROOT=$d ./testgo get -d golang.org/x/codereview/cmd/hgpatch ; then
        echo 'go get golang.org/x/codereview/cmd/hgpatch should not succeed with GOPATH=$GOROOT'
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
./testgo run -ldflags '-X main.extern "hello world"' $d/main.go 2>hello.out || ok=false
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

TEST go test -cpuprofile -o controls binary location
./testgo test -cpuprofile strings.prof -o mystrings.test strings || ok=false
if [ ! -x mystrings.test ]; then
	echo "go test -cpuprofile -o mystrings.test did not create mystrings.test"
	ok=false
fi
rm -f strings.prof mystrings.test

TEST go test -c -o controls binary location
./testgo test -c -o mystrings.test strings || ok=false
if [ ! -x mystrings.test ]; then
	echo "go test -c -o mystrings.test did not create mystrings.test"
	ok=false
fi
rm -f mystrings.test

TEST go test -o writes binary
./testgo test -o mystrings.test strings || ok=false
if [ ! -x mystrings.test ]; then
	echo "go test -o mystrings.test did not create mystrings.test"
	ok=false
fi
rm -f mystrings.test

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
./testgo get golang.org/x/tools/cmd/cover || ok=false

unset GOPATH
rm -rf $d

TEST go get -t "code.google.com/p/go-get-issue-8181/{a,b}"
d=$(TMPDIR=/var/tmp mktemp -d -t testgoXXX)
export GOPATH=$d
if ./testgo get -t code.google.com/p/go-get-issue-8181/{a,b}; then
	./testgo list ... | grep go.tools/godoc > /dev/null || ok=false
else
	ok=false
fi
unset GOPATH
rm -rf $d

TEST shadowing logic
export GOPATH=$(pwd)/testdata/shadow/root1:$(pwd)/testdata/shadow/root2

# The math in root1 is not "math" because the standard math is.
set +e
cdir=$(./testgo list -f '({{.ImportPath}}) ({{.ConflictDir}})' ./testdata/shadow/root1/src/math)
set -e
if [ "$cdir" != "(_$(pwd)/testdata/shadow/root1/src/math) ($GOROOT/src/math)" ]; then
	echo shadowed math is not shadowed: "$cdir"
	ok=false
fi

# The foo in root1 is "foo".
set +e
cdir=$(./testgo list -f '({{.ImportPath}}) ({{.ConflictDir}})' ./testdata/shadow/root1/src/foo)
set -e
if [ "$cdir" != "(foo) ()" ]; then
	echo unshadowed foo is shadowed: "$cdir"
	ok=false
fi

# The foo in root2 is not "foo" because the foo in root1 got there first.
set +e
cdir=$(./testgo list -f '({{.ImportPath}}) ({{.ConflictDir}})' ./testdata/shadow/root2/src/foo)
set -e
if [ "$cdir" != "(_$(pwd)/testdata/shadow/root2/src/foo) ($(pwd)/testdata/shadow/root1/src/foo)" ]; then
	echo shadowed foo is not shadowed: "$cdir"
	ok=false
fi

# The error for go install should mention the conflicting directory.
set +e
err=$(./testgo install ./testdata/shadow/root2/src/foo 2>&1)
set -e
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
checkbar() {
	desc="$1"
	sleep 1
	touch $d/src/x/y/foo/foo.go
	if ! ./testgo build -v -i x/y/bar &> $d/err; then
		echo build -i "$1" failed
		cat $d/err
		ok=false
	elif ! grep x/y/foo $d/err >/dev/null; then
		echo first build -i "$1" did not build x/y/foo
		cat $d/err
		ok=false
	fi
	if ! ./testgo build -v -i x/y/bar &> $d/err; then
		echo second build -i "$1" failed
		cat $d/err
		ok=false
	elif grep x/y/foo $d/err >/dev/null; then
		echo second build -i "$1" built x/y/foo
		cat $d/err
		ok=false
	fi
}

echo '
package bar
import "x/y/foo"
func F() { foo.F() }
' >$d/src/x/y/bar/bar.go
checkbar pkg

TEST build -i installs dependencies for command
echo '
package main
import "x/y/foo"
func main() { foo.F() }
' >$d/src/x/y/bar/bar.go
checkbar cmd

rm -rf $d bar
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
./testgo clean -i xtestonly || ok=false
if ! ./testgo test xtestonly >/dev/null; then
	echo "go test xtestonly failed"
	ok=false
fi
unset GOPATH

TEST 'go test builds an xtest containing only non-runnable examples'
if ! ./testgo test -v ./testdata/norunexample > testdata/std.out; then
	echo "go test ./testdata/norunexample failed"
	ok=false
elif ! grep 'File with non-runnable example was built.' testdata/std.out > /dev/null; then
	echo "file with non-runnable example was not built"
	ok=false
fi

TEST 'go generate handles simple command'
if ! ./testgo generate ./testdata/generate/test1.go > testdata/std.out; then
	echo "go test ./testdata/generate/test1.go failed to run"
	ok=false
elif ! grep 'Success' testdata/std.out > /dev/null; then
	echo "go test ./testdata/generate/test1.go generated wrong output"
	ok=false
fi

TEST 'go generate handles command alias'
if ! ./testgo generate ./testdata/generate/test2.go > testdata/std.out; then
	echo "go test ./testdata/generate/test2.go failed to run"
	ok=false
elif ! grep 'Now is the time for all good men' testdata/std.out > /dev/null; then
	echo "go test ./testdata/generate/test2.go generated wrong output"
	ok=false
fi

TEST 'go generate variable substitution'
if ! ./testgo generate ./testdata/generate/test3.go > testdata/std.out; then
	echo "go test ./testdata/generate/test3.go failed to run"
	ok=false
elif ! grep "$GOARCH test3.go p xyzp/test3.go/123" testdata/std.out > /dev/null; then
	echo "go test ./testdata/generate/test3.go generated wrong output"
	ok=false
fi

TEST go get works with vanity wildcards
d=$(mktemp -d -t testgoXXX)
export GOPATH=$d
if ! ./testgo get -u rsc.io/pdf/...; then
	ok=false
elif [ ! -x $d/bin/pdfpasswd ]; then
	echo did not build rsc.io/pdf/pdfpasswd
	ok=false
fi
unset GOPATH
rm -rf $d

TEST go vet with external tests
d=$(mktemp -d -t testgoXXX)
export GOPATH=$(pwd)/testdata
if ./testgo vet vetpkg >$d/err 2>&1; then
	echo "go vet vetpkg passes incorrectly"
	ok=false
elif ! grep -q 'missing argument for Printf' $d/err; then
	echo "go vet vetpkg did not find missing argument for Printf"
	cat $d/err
	ok=false
fi
unset GOPATH
rm -rf $d

# clean up
if $started; then stop; fi
rm -rf testdata/bin testdata/bin1
rm -f testgo

if $allok; then
	echo PASS
else
	echo FAIL:
	echo "$testfail"
	exit 1
fi

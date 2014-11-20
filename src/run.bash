#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

eval $(go env)
export GOROOT   # the api test requires GOROOT to be set.

unset CDPATH	# in case user has it set
unset GOPATH    # we disallow local import for non-local packages, if $GOROOT happens
                # to be under $GOPATH, then some tests below will fail

# no core files, please
ulimit -c 0

# Raise soft limits to hard limits for NetBSD/OpenBSD.
# We need at least 256 files and ~300 MB of bss.
# On OS X ulimit -S -n rejects 'unlimited'.
#
# Note that ulimit -S -n may fail if ulimit -H -n is set higher than a
# non-root process is allowed to set the high limit.
# This is a system misconfiguration and should be fixed on the
# broken system, not "fixed" by ignoring the failure here.
# See longer discussion on golang.org/issue/7381. 
[ "$(ulimit -H -n)" == "unlimited" ] || ulimit -S -n $(ulimit -H -n)
[ "$(ulimit -H -d)" == "unlimited" ] || ulimit -S -d $(ulimit -H -d)

# Thread count limit on NetBSD 7.
if ulimit -T &> /dev/null; then
	[ "$(ulimit -H -T)" == "unlimited" ] || ulimit -S -T $(ulimit -H -T)
fi

# allow all.bash to avoid double-build of everything
rebuild=true
if [ "$1" == "--no-rebuild" ]; then
	shift
else
	echo '# Building packages and commands.'
	time go install -a -v std
	echo
fi

# we must unset GOROOT_FINAL before tests, because runtime/debug requires
# correct access to source code, so if we have GOROOT_FINAL in effect,
# at least runtime/debug test will fail.
unset GOROOT_FINAL

# increase timeout for ARM up to 3 times the normal value
timeout_scale=1
[ "$GOARCH" == "arm" ] && timeout_scale=3

echo '# Testing packages.'
time go test std -short -timeout=$(expr 120 \* $timeout_scale)s -gcflags "$GO_GCFLAGS"
echo

# We set GOMAXPROCS=2 in addition to -cpu=1,2,4 in order to test runtime bootstrap code,
# creation of first goroutines and first garbage collections in the parallel setting.
echo '# GOMAXPROCS=2 runtime -cpu=1,2,4'
GOMAXPROCS=2 go test runtime -short -timeout=$(expr 300 \* $timeout_scale)s -cpu=1,2,4
echo

echo '# sync -cpu=10'
go test sync -short -timeout=$(expr 120 \* $timeout_scale)s -cpu=10

xcd() {
	echo
	echo '#' $1
	builtin cd "$GOROOT"/src/$1 || exit 1
}

# NOTE: "set -e" cannot help us in subshells. It works until you test it with ||.
#
#	$ bash --version
#	GNU bash, version 3.2.48(1)-release (x86_64-apple-darwin12)
#	Copyright (C) 2007 Free Software Foundation, Inc.
#
#	$ set -e; (set -e; false; echo still here); echo subshell exit status $?
#	subshell exit status 1
#	# subshell stopped early, set exit status, but outer set -e didn't stop.
#
#	$ set -e; (set -e; false; echo still here) || echo stopped
#	still here
#	# somehow the '|| echo stopped' broke the inner set -e.
#	
# To avoid this bug, every command in a subshell should have '|| exit 1' on it.
# Strictly speaking, the test may be unnecessary on the final command of
# the subshell, but it aids later editing and may avoid future bash bugs.

if [ "$GOOS" == "android" ]; then
	# Disable cgo tests on android.
	# They are not designed to run off the host.
	# golang.org/issue/8345
	CGO_ENABLED=0
fi

[ "$CGO_ENABLED" != 1 ] ||
[ "$GOHOSTOS" == windows ] ||
(xcd ../misc/cgo/stdio
go run $GOROOT/test/run.go - . || exit 1
) || exit $?

[ "$CGO_ENABLED" != 1 ] ||
(xcd ../misc/cgo/life
go run $GOROOT/test/run.go - . || exit 1
) || exit $?

[ "$CGO_ENABLED" != 1 ] ||
(xcd ../misc/cgo/test
# cgo tests inspect the traceback for runtime functions
extlink=0
export GOTRACEBACK=2
go test -ldflags '-linkmode=auto' || exit 1
# linkmode=internal fails on dragonfly since errno is a TLS relocation.
[ "$GOHOSTOS" == dragonfly ] || go test -ldflags '-linkmode=internal' || exit 1
case "$GOHOSTOS-$GOARCH" in
openbsd-386 | openbsd-amd64)
	# test linkmode=external, but __thread not supported, so skip testtls.
	go test -ldflags '-linkmode=external' || exit 1
	extlink=1
	;;
darwin-386 | darwin-amd64)
	# linkmode=external fails on OS X 10.6 and earlier == Darwin
	# 10.8 and earlier.
	case $(uname -r) in
	[0-9].* | 10.*) ;;
	*)
		go test -ldflags '-linkmode=external'  || exit 1
		extlink=1
		;;
	esac
	;;
android-arm | dragonfly-386 | dragonfly-amd64 | freebsd-386 | freebsd-amd64 | freebsd-arm | linux-386 | linux-amd64 | linux-arm | netbsd-386 | netbsd-amd64)
	go test -ldflags '-linkmode=external' || exit 1
	go test -ldflags '-linkmode=auto' ../testtls || exit 1
	go test -ldflags '-linkmode=external' ../testtls || exit 1
	extlink=1
	
	case "$GOHOSTOS-$GOARCH" in
	netbsd-386 | netbsd-amd64) ;; # no static linking
	freebsd-arm) ;; # -fPIC compiled tls code will use __tls_get_addr instead
	                # of __aeabi_read_tp, however, on FreeBSD/ARM, __tls_get_addr
	                # is implemented in rtld-elf, so -fPIC isn't compatible with
	                # static linking on FreeBSD/ARM with clang. (cgo depends on
			# -fPIC fundamentally.)
	*)
		if ! $CC -xc -o /dev/null -static - 2>/dev/null <<<'int main() {}' ; then
			echo "No support for static linking found (lacks libc.a?), skip cgo static linking test."
		else
			go test -ldflags '-linkmode=external -extldflags "-static -pthread"' ../testtls || exit 1
			go test ../nocgo || exit 1
			go test -ldflags '-linkmode=external' ../nocgo || exit 1
			go test -ldflags '-linkmode=external -extldflags "-static -pthread"' ../nocgo || exit 1
		fi
		;;
	esac
	;;
esac
) || exit $?

# Race detector only supported on Linux, FreeBSD and OS X,
# and only on amd64, and only when cgo is enabled.
# Delayed until here so we know whether to try external linking.
case "$GOHOSTOS-$GOOS-$GOARCH-$CGO_ENABLED" in
linux-linux-amd64-1 | freebsd-freebsd-amd64-1 | darwin-darwin-amd64-1)
	echo
	echo '# Testing race detector.'
	go test -race -i runtime/race flag os/exec
	go test -race -run=Output runtime/race
	go test -race -short flag os/exec
	
	# Test with external linking; see issue 9133.
	if [ "$extlink" = 1 ]; then
		go test -race -short -ldflags=-linkmode=external flag os/exec
	fi
esac

# This tests cgo -cdefs. That mode is not supported,
# so it's okay if it doesn't work on some systems.
# In particular, it works badly with clang on OS X.
# It doesn't work at all now that we disallow C code
# outside runtime. Once runtime has no C code it won't
# even be necessary.
# [ "$CGO_ENABLED" != 1 ] || [ "$GOOS" == darwin ] ||
# (xcd ../misc/cgo/testcdefs
# ./test.bash || exit 1
# ) || exit $?

[ "$CGO_ENABLED" != 1 ] || [ "$GOOS" == darwin ] ||
(xcd ../misc/cgo/testgodefs
./test.bash || exit 1
) || exit $?

[ "$CGO_ENABLED" != 1 ] ||
[ "$GOHOSTOS" == windows ] ||
(xcd ../misc/cgo/testso
./test.bash || exit 1
) || exit $?

[ "$CGO_ENABLED" != 1 ] ||
[ "$GOHOSTOS-$GOARCH" != linux-amd64 ] ||
(xcd ../misc/cgo/testasan
go run main.go || exit 1
) || exit $?

[ "$CGO_ENABLED" != 1 ] ||
[ "$GOHOSTOS" == windows ] ||
(xcd ../misc/cgo/errors
./test.bash || exit 1
) || exit $?

[ "$GOOS" == nacl ] ||
[ "$GOOS" == android ] ||
(xcd ../doc/progs
time ./run || exit 1
) || exit $?

[ "$GOOS" == android ] ||
[ "$GOOS" == nacl ] ||
[ "$GOARCH" == arm ] ||  # uses network, fails under QEMU
(xcd ../doc/articles/wiki
./test.bash || exit 1
) || exit $?

[ "$GOOS" == android ] ||
[ "$GOOS" == nacl ] ||
(xcd ../doc/codewalk
time ./run || exit 1
) || exit $?

[ "$GOOS" == nacl ] ||
[ "$GOARCH" == arm ] ||
(xcd ../test/bench/shootout
time ./timing.sh -test || exit 1
) || exit $?

[ "$GOOS" == android ] || # TODO(crawshaw): get this working
[ "$GOOS" == openbsd ] || # golang.org/issue/5057
(
echo
echo '#' ../test/bench/go1
go test ../test/bench/go1 || exit 1
) || exit $?

[ "$GOOS" == android ] ||
(xcd ../test
unset GOMAXPROCS
GOOS=$GOHOSTOS GOARCH=$GOHOSTARCH go build -o runtest run.go || exit 1
time ./runtest || exit 1
rm -f runtest
) || exit $?

[ "$GOOS" == android ] ||
[ "$GOOS" == nacl ] ||
(
echo
echo '# Checking API compatibility.'
time go run $GOROOT/src/cmd/api/run.go || exit 1
) || exit $?

echo
echo ALL TESTS PASSED

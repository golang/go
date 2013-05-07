#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

eval $(go env)

unset CDPATH	# in case user has it set
unset GOPATH    # we disallow local import for non-local packages, if $GOROOT happens
                # to be under $GOPATH, then some tests below will fail

# no core files, please
ulimit -c 0

# Raise soft limits to hard limits for NetBSD/OpenBSD.
# We need at least 256 files and ~300 MB of bss.
# On OS X ulimit -S -n rejects 'unlimited'.
[ "$(ulimit -H -n)" == "unlimited" ] || ulimit -S -n $(ulimit -H -n)
[ "$(ulimit -H -d)" == "unlimited" ] || ulimit -S -d $(ulimit -H -d)

# allow all.bash to avoid double-build of everything
rebuild=true
if [ "$1" = "--no-rebuild" ]; then
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
time go test std -short -timeout=$(expr 120 \* $timeout_scale)s
echo

echo '# GOMAXPROCS=2 runtime -cpu=1,2,4'
GOMAXPROCS=2 go test runtime -short -timeout=$(expr 240 \* $timeout_scale)s -cpu=1,2,4
echo

echo '# sync -cpu=10'
go test sync -short -timeout=$(expr 120 \* $timeout_scale)s -cpu=10

# Race detector only supported on Linux and OS X,
# and only on amd64, and only when cgo is enabled.
case "$GOHOSTOS-$GOOS-$GOARCH-$CGO_ENABLED" in
linux-linux-amd64-1 | darwin-darwin-amd64-1)
	echo
	echo '# Testing race detector.'
	go test -race -i flag
	go test -race -short flag
esac

xcd() {
	echo
	echo '#' $1
	builtin cd "$GOROOT"/src/$1
}

[ "$CGO_ENABLED" != 1 ] ||
[ "$GOHOSTOS" == windows ] ||
(xcd ../misc/cgo/stdio
go run $GOROOT/test/run.go - .
) || exit $?

[ "$CGO_ENABLED" != 1 ] ||
(xcd ../misc/cgo/life
go run $GOROOT/test/run.go - .
) || exit $?

[ "$CGO_ENABLED" != 1 ] ||
(xcd ../misc/cgo/test
set -e
go test -ldflags '-linkmode=auto'
go test -ldflags '-linkmode=internal'
case "$GOHOSTOS-$GOARCH" in
openbsd-386 | openbsd-amd64)
	# test linkmode=external, but __thread not supported, so skip testtls.
	go test -ldflags '-linkmode=external'
	;;
darwin-386 | darwin-amd64)
	# linkmode=external fails on OS X 10.6 and earlier == Darwin
	# 10.8 and earlier.
	case $(uname -r) in
	[0-9].* | 10.*) ;;
	*) go test -ldflags '-linkmode=external' ;;
	esac
	;;
freebsd-386 | freebsd-amd64 | linux-386 | linux-amd64 | netbsd-386 | netbsd-amd64)
	go test -ldflags '-linkmode=external'
	go test -ldflags '-linkmode=auto' ../testtls
	go test -ldflags '-linkmode=external' ../testtls
esac
) || exit $?

[ "$CGO_ENABLED" != 1 ] ||
[ "$GOHOSTOS" == windows ] ||
[ "$GOHOSTOS" == darwin ] ||
(xcd ../misc/cgo/testso
./test.bash
) || exit $?

(xcd ../doc/progs
time ./run
) || exit $?

[ "$GOARCH" == arm ] ||  # uses network, fails under QEMU
(xcd ../doc/articles/wiki
make clean
./test.bash
) || exit $?

(xcd ../doc/codewalk
# TODO: test these too.
set -e
go build pig.go
go build urlpoll.go
rm -f pig urlpoll
) || exit $?

echo
echo '#' ../misc/dashboard/builder ../misc/goplay
go build ../misc/dashboard/builder ../misc/goplay

[ "$GOARCH" == arm ] ||
(xcd ../test/bench/shootout
./timing.sh -test
) || exit $?

[ "$GOOS" == openbsd ] || # golang.org/issue/5057
(
echo
echo '#' ../test/bench/go1
go test ../test/bench/go1
) || exit $?

(xcd ../test
unset GOMAXPROCS
time go run run.go
) || exit $?

echo
echo '# Checking API compatibility.'
go tool api -c $GOROOT/api/go1.txt,$GOROOT/api/go1.1.txt -next $GOROOT/api/next.txt -except $GOROOT/api/except.txt

echo
echo ALL TESTS PASSED

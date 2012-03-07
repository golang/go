#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

eval $(go tool dist env)

unset CDPATH	# in case user has it set

# no core files, please
ulimit -c 0

# allow all.bash to avoid double-build of everything
rebuild=true
if [ "$1" = "--no-rebuild" ]; then
	shift
else
	echo '# Building packages and commands.'
	time go install -a -v std
	echo
fi

echo '# Testing packages.'
time go test std -short -timeout=120s
echo

echo '# GOMAXPROCS=2 runtime -cpu=1,2,4'
GOMAXPROCS=2 go test runtime -short -timeout=120s -cpu=1,2,4
echo

echo '# sync -cpu=10'
go test sync -short -timeout=120s -cpu=10

xcd() {
	echo
	echo '#' $1
	builtin cd "$GOROOT"/src/$1
}

[ "$CGO_ENABLED" != 1 ] ||
[ "$GOHOSTOS" == windows ] ||
(xcd ../misc/cgo/stdio
./test.bash
) || exit $?

[ "$CGO_ENABLED" != 1 ] ||
(xcd ../misc/cgo/life
./test.bash
) || exit $?

[ "$CGO_ENABLED" != 1 ] ||
(xcd ../misc/cgo/test
go test
) || exit $?

$BROKEN ||
[ "$CGO_ENABLED" != 1 ] ||
[ "$GOHOSTOS" == windows ] ||
[ "$GOHOSTOS" == darwin ] ||
(xcd ../misc/cgo/testso
"$GOMAKE" clean
./test.bash
) || exit $?

(xcd ../doc/progs
time ./run
) || exit $?

[ "$GOARCH" == arm ] ||  # uses network, fails under QEMU
(xcd ../doc/articles/wiki
make clean
./test.sh
) || exit $?

echo
echo '#' ../misc/dashboard/builder ../misc/goplay
go build ../misc/dashboard/builder ../misc/goplay || exit $?

[ "$GOARCH" == arm ] ||
(xcd ../test/bench/shootout
./timing.sh -test
) || exit $?

echo
echo '#' ../test/bench/go1
go test ../test/bench/go1 || exit $?

(xcd ../test
time go run run.go
) || exit $?

echo
echo ALL TESTS PASSED

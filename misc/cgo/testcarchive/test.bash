#!/usr/bin/env bash
# Copyright 2015 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

ccargs=
if [ "$(go env GOOS)" == "darwin" ]; then
	ccargs="-Wl,-no_pie"
	# For darwin/arm.
	# TODO(crawshaw): Can we do better?
	ccargs="$ccargs -framework CoreFoundation -framework Foundation"
fi
ccargs="$ccargs -I pkg/$(go env GOOS)_$(go env GOARCH)"

# TODO(crawshaw): Consider a go env for exec script name.
bin=./testp
exec_script=go_$(go env GOOS)_$(go env GOARCH)_exec
if [ "$(which $exec_script)" != "" ]; then
	bin="$exec_script ./testp"
fi

rm -rf libgo.a libgo.h testp pkg

status=0

# Installing first will create the header files we want.

GOPATH=$(pwd) go install -buildmode=c-archive libgo
$(go env CC) $(go env GOGCCFLAGS) $ccargs -o testp main.c pkg/$(go env GOOS)_$(go env GOARCH)/libgo.a
if ! $bin arg1 arg2; then
    echo "FAIL test1a"
    status=1
fi
rm -f libgo.a libgo.h testp

# Test building libgo other than installing it.
# Header files are now present.

GOPATH=$(pwd) go build -buildmode=c-archive src/libgo/libgo.go
$(go env CC) $(go env GOGCCFLAGS) $ccargs -o testp main.c libgo.a
if ! $bin arg1 arg2; then
    echo "FAIL test1b"
    status=1
fi
rm -f libgo.a libgo.h testp

GOPATH=$(pwd) go build -buildmode=c-archive -o libgo.a libgo
$(go env CC) $(go env GOGCCFLAGS) $ccargs -o testp main.c libgo.a
if ! $bin arg1 arg2; then
    echo "FAIL test1c"
    status=1
fi
rm -rf libgo.a libgo.h testp pkg

case "$(go env GOOS)/$(go env GOARCH)" in
"darwin/arm" | "darwin/arm64")
    echo "Skipping test2; see https://golang.org/issue/13701"
    ;;
*)
    GOPATH=$(pwd) go build -buildmode=c-archive -o libgo2.a libgo2
    $(go env CC) $(go env GOGCCFLAGS) $ccargs -o testp main2.c libgo2.a
    if ! $bin; then
        echo "FAIL test2"
        status=1
    fi
    rm -rf libgo2.a libgo2.h testp pkg
    ;;
esac

GOPATH=$(pwd) go build -buildmode=c-archive -o libgo3.a libgo3
$(go env CC) $(go env GOGCCFLAGS) $ccargs -o testp main3.c libgo3.a
if ! $bin; then
    echo "FAIL test3"
    status=1
fi
rm -rf libgo3.a libgo3.h testp pkg

GOPATH=$(pwd) go build -buildmode=c-archive -o libgo4.a libgo4
$(go env CC) $(go env GOGCCFLAGS) $ccargs -o testp main4.c libgo4.a
if ! $bin; then
    echo "FAIL test4"
    status=1
fi
rm -rf libgo4.a libgo4.h testp pkg

exit $status

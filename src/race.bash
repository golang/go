#!/usr/bin/env bash
# Copyright 2013 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# race.bash tests the standard library under the race detector.
# https://golang.org/doc/articles/race_detector.html

set -e

function usage {
	echo 'race detector is only supported on linux/amd64, linux/ppc64le, linux/arm64, freebsd/amd64, netbsd/amd64 and darwin/amd64' 1>&2
	exit 1
}

case $(uname) in
"Darwin")
	# why Apple? why?
	if sysctl machdep.cpu.extfeatures | grep -qv EM64T; then
		usage
	fi
	;;
"Linux")
	if [ $(uname -m) != "x86_64" ] && [ $(uname -m) != "ppc64le" ] && [ $(uname -m) != "aarch64" ]; then
		usage
	fi
	;;
"FreeBSD")
	if [ $(uname -m) != "amd64" ]; then
		usage
	fi
	;;
"NetBSD")
	if [ $(uname -m) != "amd64" ]; then
		usage
	fi
	;;
*)
	usage
	;;
esac

if [ ! -f make.bash ]; then
	echo 'race.bash must be run from $GOROOT/src' 1>&2
	exit 1
fi
. ./make.bash --no-banner
go install -race std
go tool dist test -race

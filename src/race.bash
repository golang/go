#!/usr/bin/env bash
# Copyright 2013 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# race.bash tests the standard library under the race detector.
# https://golang.org/doc/articles/race_detector.html

set -e

function usage {
	echo 'race detector is only supported on linux/amd64, linux/ppc64le, linux/arm64, linux/loong64, linux/riscv64, linux/s390x, freebsd/amd64, netbsd/amd64, openbsd/amd64, darwin/amd64, and darwin/arm64' 1>&2
	exit 1
}

case $(uname -s -m) in
  "Darwin x86_64") ;;
  "Darwin arm64")  ;;
  "Linux x86_64")  ;;
  "Linux ppc64le") ;;
  "Linux aarch64") ;;
  "Linux loongarch64") ;;
  "Linux riscv64") ;;
  "Linux s390x")   ;;
  "FreeBSD amd64") ;;
  "NetBSD amd64")  ;;
  "OpenBSD amd64") ;;
  *) usage         ;;
esac

if [ ! -f make.bash ]; then
	echo 'race.bash must be run from $GOROOT/src' 1>&2
	exit 1
fi
. ./make.bash --no-banner
go install -race std
go tool dist test -race

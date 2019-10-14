#!/usr/bin/env bash
# Copyright 2016 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This directory is intended to test the use of Fortran with cgo.

set -e

FC=$1

goos=$(go env GOOS)

libext="so"
if [ "$goos" = "darwin" ]; then
	libext="dylib"
elif [ "$goos" = "aix" ]; then
	libtext="a"
fi

case "$FC" in
*gfortran*)
  libpath=$(dirname $($FC -print-file-name=libgfortran.$libext))
  if [ "$goos" != "aix" ]; then
	  RPATH_FLAG="-Wl,-rpath,$libpath"
  fi
  export CGO_LDFLAGS="$CGO_LDFLAGS $RPATH_FLAG -L $libpath"
  ;;
esac

if ! $FC helloworld/helloworld.f90 -o main.exe >& /dev/null; then
  echo "skipping Fortran test: could not build helloworld.f90 with $FC"
  exit 0
fi
rm -f main.exe

status=0

if ! go test; then
  echo "FAIL: go test"
  status=1
fi

exit $status

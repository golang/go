#!/usr/bin/env bash
# Copyright 2011 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

eval $(gomake --no-print-directory -f ../../Make.inc go-env)

TMP="testdata_tmp.go"
TEST="testdata.c"
GOLDEN="testdata_${GOOS}_${GOARCH}.golden"

case ${GOARCH} in
"amd64") CCARG="-f-m64";;
"386") CCARG="-f-m32";;
*) CCARG="";;
esac

cleanup() {
	rm ${TMP}
}

error() {
	cleanup
	echo $1
	exit 1
}

if [ ! -e ${GOLDEN} ]; then
	echo "skipping - no golden defined for this platform"
	exit
fi

./godefs -g test ${CCARG} ${TEST} > ${TMP}
if [ $? != 0 ]; then
	error "Error: Could not run godefs for ${TEST}"
fi

diff ${TMP} ${GOLDEN}
if [ $? != 0 ]; then
	error "FAIL: godefs for ${TEST} did not match ${GOLDEN}"
fi

cleanup

echo "PASS"

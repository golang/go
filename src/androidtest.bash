#!/usr/bin/env bash
# Copyright 2014 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# For testing Android.
# The compiler runs locally, then a copy of the GOROOT is pushed to a
# target device using adb, and the tests are run there.

set -e
ulimit -c 0 # no core files

if [ ! -f make.bash ]; then
	echo 'androidtest.bash must be run from $GOROOT/src' 1>&2
	exit 1
fi

if [ -z $GOOS ]; then
	export GOOS=android
fi
if [ "$GOOS" != "android" ]; then
	echo "androidtest.bash requires GOOS=android, got GOOS=$GOOS" 1>&2
	exit 1
fi

export CGO_ENABLED=1

# Run the build for the host bootstrap, so we can build go_android_exec.
# Also lets us fail early before the (slow) adb push if the build is broken.
./make.bash
export GOROOT=$(dirname $(pwd))
export PATH=$GOROOT/bin:$PATH
GOOS=$GOHOSTOS GOARCH=$GOHOSTARCH go build \
	-o ../bin/go_android_${GOARCH}_exec \
	../misc/android/go_android_exec.go

# Push GOROOT to target device.
#
# The adb sync command will sync either the /system or /data
# directories of an android device from a similar directory
# on the host. We copy the files required for running tests under
# /data/local/tmp/goroot. The adb sync command does not follow
# symlinks so we have to copy.
export ANDROID_PRODUCT_OUT=/tmp/androidtest-$$
FAKE_GOROOT=$ANDROID_PRODUCT_OUT/data/local/tmp/goroot
mkdir -p $FAKE_GOROOT
cp -R --preserve=all "${GOROOT}/src" "${FAKE_GOROOT}/"
cp -R --preserve=all "${GOROOT}/test" "${FAKE_GOROOT}/"
cp -R --preserve=all "${GOROOT}/lib" "${FAKE_GOROOT}/"
echo '# Syncing test files to android device'
time adb sync data &> /dev/null
echo ''
rm -rf "$ANDROID_PRODUCT_OUT"

# Run standard build and tests.
./all.bash --no-clean

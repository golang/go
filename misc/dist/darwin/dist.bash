#!/bin/bash
# Copyright 2011 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

if ! test -f ../../../src/all.bash; then
	echo >&2 "dist.bash must be run from $GOROOT/misc/dist/darwin"
	exit 1
fi

echo >&2 "Locating PackageMaker..."
PM=/Applications/Utilities/PackageMaker.app/Contents/MacOS/PackageMaker
if [ ! -x $PM ]; then
	PM=/Developer$PM
	if [ ! -x $PM ]; then
		echo >&2 "could not find PackageMaker; aborting"
		exit 1
	fi
fi
echo >&2 "  Found: $PM"

BUILD=/tmp/go.build.tmp
ROOT=`hg root`
export GOROOT=$BUILD/root/usr/local/go
export GOROOT_FINAL=/usr/local/go

echo >&2 "Removing old images"
rm -f *.pkg *.dmg

echo >&2 "Preparing temporary directory"
rm -rf $BUILD
mkdir -p $BUILD
trap "rm -rf $BUILD" 0

echo >&2 "Copying go source distribution"
mkdir -p $BUILD/root/usr/local
cp -r $ROOT $GOROOT
cp -r etc $BUILD/root/etc

pushd $GOROOT > /dev/null

echo >&2 "Detecting version..."
pushd src > /dev/null
./make.bash --dist-tool > /dev/null
../bin/tool/dist version > /dev/null
popd > /dev/null
mv VERSION.cache VERSION
VERSION="$(cat VERSION | awk '{ print $1 }')"
echo >&2 "  Version: $VERSION"

echo >&2 "Pruning Mercurial metadata"
rm -rf .hg .hgignore .hgtags

echo >&2 "Building Go"
pushd src
./all.bash 2>&1 | sed "s/^/  /" >&2
popd > /dev/null

popd > /dev/null

echo >&2 "Building package"
$PM -v -r $BUILD/root -o "go.darwin.$VERSION.pkg" \
	--scripts scripts \
	--id com.googlecode.go \
	--title Go \
	--version "0.1" \
	--target "10.5"

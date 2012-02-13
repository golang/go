#!/usr/bin/env bash
# Copyright 2012 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

TAG=$1
if [ "$TAG" == "" ]; then
	echo >&2 'usage: dist.bash <tag>'
	exit 2
fi

GOOS=${GOOS:-linux}
GOARCH=${GOARCH:-amd64}

ROOT=/tmp/godist.linux.$GOARCH
rm -rf $ROOT
mkdir -p $ROOT
pushd $ROOT>/dev/null

# clone Go distribution
echo "Preparing new GOROOT"
hg clone -q https://code.google.com/p/go go
pushd go > /dev/null
hg update $TAG

# get version
pushd src > /dev/null
echo "Building dist tool to get VERSION"
./make.bash --dist-tool 2>&1 | sed 's/^/  /' >&2
../bin/tool/dist version > ../VERSION
popd > /dev/null
VERSION="$(cat VERSION | awk '{ print $1 }')"
echo "  Version: $VERSION"

# remove mercurial stuff
rm -rf .hg*

# build Go
echo "Building Go"
unset GOROOT
export GOOS
export GOARCH
export GOROOT_FINAL=/usr/local/go
pushd src > /dev/null
./all.bash 2>&1 | sed 's/^/  /' >&2
popd > /dev/null
popd > /dev/null

# tar it up
DEST=go.$VERSION.$GOOS-$GOARCH.tar.gz
echo "Writing tarball: $ROOT/$DEST"
tar czf $DEST go
popd > /dev/null

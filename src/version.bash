#!/usr/bin/env bash
# Copyright 2010 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

GOROOT=$(dirname $0)/..

# If a version file created by -save is available, use it
if [ -f "$GOROOT/VERSION" -a "$1" != "-save" ]; then
	cat $GOROOT/VERSION
	exit 0
fi

# Otherwise, if hg doesn't work for whatever reason, fail
if [ ! -d "$GOROOT/.hg" ] || ! hg version > /dev/null 2>&1; then
	echo 'Unable to report version: hg and VERSION file missing' 1>&2
	echo 'Generate VERSION with `src/version.bash -save` while hg is usable' 1>&2
	exit 2
fi

# Get numerical revision
VERSION=$(hg identify -n 2>/dev/null)
if [ $? != 0 ]; then
	OLD=$(hg identify | sed 1q)
	VERSION=$(echo $OLD | awk '{print $1}')
fi

# Get branch type
BRANCH=release
if [ "$(hg identify -b 2>/dev/null)" == "default" ]; then
	BRANCH=weekly
fi

# Find most recent known release or weekly tag.
TAG=$(hg tags |
	grep $BRANCH |
	sed 's/:.*//' |
	sort -rn -k2 |
	awk -v ver=$VERSION '$2+0 <= ver+0 && $1~/^(release|weekly)\./ {print $1}' |
	sed -n 1p)

if [ "$TAG" != "" ]; then
	VERSION="$TAG $VERSION"
fi

if [ "$1" = "-save" ]; then
	echo $VERSION > $GOROOT/VERSION
else
	echo $VERSION
fi

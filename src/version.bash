#!/usr/bin/env bash
# Copyright 2010 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Check that we can use 'hg'
if ! hg version > /dev/null 2>&1; then
	echo 'unable to report version: hg not installed' 1>&2
	echo 'unknown'
	exit 0
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
	awk -v ver=$VERSION '$2 <= ver && $1~/^(release|weekly)\./ {print $1}' |
	sed -n 1p)

if [ "$TAG" != "" ]; then
	VERSION="$TAG $VERSION"
fi

echo $VERSION


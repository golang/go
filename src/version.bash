#!/usr/bin/env bash
# Copyright 2010 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Check that we can use 'hg'
if ! hg version > /dev/null 2>&1; then
	echo 'hg not installed' 1>&2
	exit 2
fi

# Get numerical revision
VERSION=$(hg identify -n 2>/dev/null)
if [ $? != 0 ]; then
	OLD=$(hg identify | sed 1q)
	VERSION=$(echo $OLD | awk '{print $1}')
fi

# Find most recent known release tag.
TAG=$(hg tags | awk '$1~/^release\./ {print $1}' | sed -n 1p)

if [ "$TAG" != "" ]; then
	VERSION="$TAG $VERSION"
fi

echo $VERSION


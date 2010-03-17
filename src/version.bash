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
if [ $? = 0 ]; then
	TAG=$(hg identify -t | sed 's!/release!!')
else
	OLD=$(hg identify | sed 1q)
	VERSION=$(echo $OLD | awk '{print $1}')
	TAG=$(echo $OLD | awk '{print $2}' | sed 's!/release!!')
fi

# Append tag if not 'tip'
if [[ "$TAG" != "tip" ]]; then
	VERSION="$VERSION $TAG"
fi

echo $VERSION


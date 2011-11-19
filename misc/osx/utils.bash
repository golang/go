#!/bin/bash
# Copyright 2011 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

echo "Attempting to locate needed utilities..."

# PackageMaker
PM=/Applications/Utilities/PackageMaker.app/Contents/MacOS/PackageMaker
if [ ! -x ${PM} ]; then
	PM=/Developer${PM}
	if [ ! -x ${PM} ]; then
		echo "Could not find PackageMaker; aborting!"
	fi
fi
echo "  PackageMaker : ${PM}"

# hdiutil. If this doesn't exist, your OS X installation is horribly borked,
# but let's check anyway...
if which hdiutil > /dev/null; then
	HDIUTIL=`which hdiutil`
	echo "  hdiutil      : ${HDIUTIL}"
fi

# Ditto for osascript
if which osascript > /dev/null; then
	OSASCRIPT=`which osascript`
	echo "  osascript    : ${OSASCRIPT}"
fi

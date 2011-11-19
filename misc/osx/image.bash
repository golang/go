#!/bin/bash
# Copyright 2011 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

source utils.bash

if ! test -f ../../src/env.bash; then
	echo "package.bash must be run from $GOROOT/misc/osx" 1>&2
fi

ROOT=`hg root`

echo "Running package.bash"
./package.bash

echo "Preparing image directory"
IMGDIR=/tmp/"Go `hg id`"
rm -rf "${IMGDIR}"
mkdir -p "${IMGDIR}"

# Copy in files
cp "Go `hg id`.pkg" "${IMGDIR}/Go.pkg"
cp ${ROOT}/LICENSE "${IMGDIR}/License.txt"
cp ReadMe.txt "${IMGDIR}/ReadMe.txt"
cp "${ROOT}/doc/gopher/bumper640x360.png" "${IMGDIR}/.background"

# Call out to applescript (osascript) to prettify things
#${OSASCRIPT} prepare.applescript

echo "Creating dmg"
${HDIUTIL} create -srcfolder "${IMGDIR}" "Go `hg id`.dmg"

echo "Removing image directory"
rm -rf ${IMGDIR}


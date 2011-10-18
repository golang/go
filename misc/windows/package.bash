#!/usr/bin/env bash
# Copyright 2010 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
set -e

ISCC="C:/Program Files/Inno Setup 5/ISCC.exe"

echo "%%%%% Checking for Inno Setup %%%%%" 1>&2
if ! test -f "$ISCC"; then
	ISCC="C:/Program Files (x86)/Inno Setup 5/ISCC.exe"
	if ! test -f "$ISCC"; then
		echo "No Inno Setup installation found" 1>&2
		exit 1
	fi
fi

echo "%%%%% Checking the packager's path %%%%%" 1>&2
if ! test -f ../../src/env.bash; then
	echo "package.bash must be run from $GOROOT/misc/windows" 1>&2
fi

echo "%%%%% Setting the go package version info %%%%%" 1>&2
ver="$(bash ../../src/version.bash | sed 's/ .*//')"

rm -rf go
mkdir go

echo "%%%%% Cloning the go tree %%%%%" 1>&2
hg clone -r $(hg id -n | sed 's/+//') $(hg root) go

rm -rf ./go/.hg ./go/.hgignore ./go/.hgtags

echo "%%%%% Copying pkg and bin %%%%%" 1>&2
cp -a ../../pkg go/pkg
cp -a ../../bin go/bin

echo "%%%%% Starting zip packaging %%%%%" 1>&2
7za a -tzip -mx=9 gowin386"_"$ver.zip "go/" >/dev/null

echo "%%%%% Starting installer packaging %%%%%" 1>&2
"$ISCC" //dAppName=Go //dAppVersion=386"_"$ver //dAppNameLower=go installer.iss  >/dev/null



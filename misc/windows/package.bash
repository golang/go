#!/usr/bin/env bash
# Copyright 2011 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
set -e

PROGS="
	candle
	light
	heat
"

echo "%%%%% Checking for WiX executables %%%%%" 1>&2
for i in $PROGS; do
	if ! which -a $1 >/dev/null; then
		echo "Cannot find '$i' on search path." 1>$2
		exit 1
	fi
done

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

echo "%%%%% Copying pkg, bin and src/pkg/runtime/z* %%%%%" 1>&2
cp -a ../../pkg go/pkg
cp -a ../../bin go/bin
cp ../../src/pkg/runtime/z*.c go/src/pkg/runtime/
cp ../../src/pkg/runtime/z*.go go/src/pkg/runtime/
cp ../../src/pkg/runtime/z*.h go/src/pkg/runtime/

echo "%%%%% Starting zip packaging %%%%%" 1>&2
7za a -tzip -mx=9 gowin$GOARCH"_"$ver.zip "go/" >/dev/null

echo "%%%%% Starting Go directory file harvesting %%%%%" 1>&2
heat dir go -nologo -cg AppFiles -gg -g1 -srd -sfrag -template fragment -dr INSTALLDIR -var var.SourceDir -out AppFiles.wxs

echo "%%%%% Starting installer packaging %%%%%" 1>&2
candle -nologo -dVersion=$ver -dArch=$GOARCH -dSourceDir=go installer.wxs AppFiles.wxs
light -nologo -ext WixUIExtension -ext WixUtilExtension installer.wixobj AppFiles.wixobj -o gowin$GOARCH"_"$ver.msi

rm -f *.wixobj AppFiles.wxs *.wixpdb


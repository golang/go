#!/bin/sh
# This uses the latest available iOS SDK, which is recommended.
# To select a specific SDK, run 'xcodebuild -showsdks'
# to see the available SDKs and replace iphoneos with one of them.
SDK=iphoneos
SDK_PATH=`xcrun --sdk $SDK --show-sdk-path`
export IPHONEOS_DEPLOYMENT_TARGET=5.1
# cmd/cgo doesn't support llvm-gcc-4.2, so we have to use clang.
CLANG=`xcrun --sdk $SDK --find clang`

if [ "$GOARCH" == "arm" ]; then
	CLANGARCH="armv7"
elif [ "$GOARCH" == "arm64" ]; then
	CLANGARCH="arm64"
else
	echo "unknown GOARCH=$GOARCH" >&2
	exit 1
fi

exec $CLANG -arch $CLANGARCH -isysroot $SDK_PATH "$@"

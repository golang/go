#!/bin/sh
# This uses the latest available iOS SDK, which is recommended.
# To select a specific SDK, run 'xcodebuild -showsdks'
# to see the available SDKs and replace iphoneos with one of them.
if [ "$GOARCH" == "arm64" ]; then
	SDK=iphoneos
	PLATFORM=ios
	CLANGARCH="arm64"
else
	SDK=iphonesimulator
	PLATFORM=ios-simulator
	CLANGARCH="x86_64"
fi

SDK_PATH=`xcrun --sdk $SDK --show-sdk-path`
export IPHONEOS_DEPLOYMENT_TARGET=5.1
# cmd/cgo doesn't support llvm-gcc-4.2, so we have to use clang.
CLANG=`xcrun --sdk $SDK --find clang`

exec "$CLANG" -arch $CLANGARCH -isysroot "$SDK_PATH" -m${PLATFORM}-version-min=12.0 "$@"

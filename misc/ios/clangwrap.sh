#!/bin/sh

# This script configures clang to target the iOS simulator. If you'd like to
# build for real iOS devices, change SDK to "iphoneos" and PLATFORM to "ios".
# This uses the latest available iOS SDK, which is recommended. To select a
# specific SDK, run 'xcodebuild -showsdks' to see the available SDKs and replace
# iphonesimulator with one of them.

SDK=iphonesimulator
PLATFORM=ios-simulator

if [ "$GOARCH" == "arm64" ]; then
	CLANGARCH="arm64"
else
	CLANGARCH="x86_64"
fi

SDK_PATH=`xcrun --sdk $SDK --show-sdk-path`

# cmd/cgo doesn't support llvm-gcc-4.2, so we have to use clang.
CLANG=`xcrun --sdk $SDK --find clang`

exec "$CLANG" -arch $CLANGARCH -isysroot "$SDK_PATH" -m${PLATFORM}-version-min=12.0 "$@"

#!/usr/bin/env bash
# Copyright 2015 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# For testing darwin/arm{,64} on iOS.

set -e
ulimit -c 0 # no core files

if [ ! -f make.bash ]; then
	echo 'iostest.bash must be run from $GOROOT/src' 1>&2
	exit 1
fi

if [ -z $GOOS ]; then
	export GOOS=darwin
fi
if [ "$GOOS" != "darwin" ]; then
	echo "iostest.bash requires GOOS=darwin, got GOOS=$GOOS" 1>&2
	exit 1
fi
if [ "$GOARCH" != "arm" ] && [ "$GOARCH" != "arm64" ]; then
	echo "iostest.bash requires GOARCH=arm or GOARCH=arm64, got GOARCH=$GOARCH" 1>&2
	exit 1
fi
if [ "$GOARCH" = "arm" ]; then
	export GOARM=7
fi

if [ "$1" = "-restart" ]; then
	# Reboot to make sure previous runs do not interfere with the current run.
	# It is reasonably easy for a bad program leave an iOS device in an
	# almost unusable state.
	idevicediagnostics restart
	# Initial sleep to make sure we are restarting before we start polling.
	sleep 30
	# Poll until the device has restarted.
	until idevicediagnostics diagnostics; do
		# TODO(crawshaw): replace with a test app using go_darwin_arm_exec.
		echo "waiting for idevice to come online"
		sleep 10
	done
	# Diagnostics are reported during boot before the device can start an
	# app. Wait a little longer before trying to use the device.
	sleep 30
fi

unset GOBIN
export GOROOT=$(dirname $(pwd))
export PATH=$GOROOT/bin:$PATH
export CGO_ENABLED=1
export CC_FOR_TARGET=$GOROOT/misc/ios/clangwrap.sh

# Run the build for the host bootstrap, so we can build go_darwin_arm_exec.
# Also lets us fail early before the (slow) ios-deploy if the build is broken.
./make.bash

GOOS=$GOHOSTOS GOARCH=$GOHOSTARCH go build \
	-o ../bin/go_darwin_${GOARCH}_exec \
	../misc/ios/go_darwin_arm_exec.go

if [ "$GOIOS_DEV_ID" = "" ]; then
	echo "detecting iOS development identity"
	eval $(GOOS=$GOHOSTOS GOARCH=$GOHOSTARCH go run ../misc/ios/detect.go)
fi

# Run standard build and tests.
./all.bash --no-clean

#!/bin/bash
# Copyright 2022 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This shell script uses Docker to run build-boring.sh and build-goboring.sh,
# which build goboringcrypto_linux_$GOARCH.syso according to the Security Policy.
# Currently, amd64 and arm64 are permitted.

set -e
set -o pipefail

GOARCH=${GOARCH:-$(go env GOARCH)}
echo "# Building goboringcrypto_linux_$GOARCH.syso. Set GOARCH to override." >&2

if ! which docker >/dev/null; then
	echo "# Docker not found. Inside Google, see go/installdocker." >&2
	exit 1
fi

platform=""
buildargs=""
case "$GOARCH" in
amd64)
	;;
arm64)
	if ! docker run --rm -t arm64v8/ubuntu:focal uname -m >/dev/null 2>&1; then
		echo "# Docker cannot run arm64 binaries. Try:"
		echo "	sudo apt-get install qemu binfmt-support qemu-user-static"
		echo "	docker run --rm --privileged multiarch/qemu-user-static --reset -p yes"
		echo "	docker run --rm -t arm64v8/ubuntu:focal uname -m"
		exit 1
	fi
	platform="--platform linux/arm64/v8"
	buildargs="--build-arg ubuntu=arm64v8/ubuntu"
	;;
*)
	echo unknown GOARCH $GOARCH >&2
	exit 2
esac

docker build $platform $buildargs --build-arg GOARCH=$GOARCH -t goboring:$GOARCH .
id=$(docker create $platform goboring:$GOARCH)
docker cp $id:/boring/godriver/goboringcrypto_linux_$GOARCH.syso ./syso
docker rm $id
ls -l ./syso/goboringcrypto_linux_$GOARCH.syso

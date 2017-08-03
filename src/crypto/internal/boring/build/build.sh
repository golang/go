#!/bin/bash
# Copyright 2017 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Run on Ubuntu system set up with:
#	sudo apt-get install debootstrap
#	sudo apt-get install squid-deb-proxy
#
# The script sets up an Ubuntu chroot and then runs the build
# in that chroot, to make sure we know exactly what software
# is being used for the build. To repeat the script reusing the
# chroot installation, run build.sh -quick. This mode is useful
# if all you've modified is goboringcrypto.c and ../goboringcrypto.h
# (or some of the setup scripts in this directory).

# Comment this setting out if not using squid-deb-proxy,
# but it will be much slower to repeat the script.
http_proxy=http://127.0.0.1:8000

chroot=/var/tmp/boringssl

sudo umount -f $chroot/proc
sudo umount -f $chroot/sys
sudo umount -f $chroot/dev/pts
sudo umount -f $chroot/dev

set -e
if [ "$1" != "-quick" ]; then
	sudo rm -rf $chroot
	sudo http_proxy=$http_proxy debootstrap --variant=minbase zesty $chroot
fi

sudo chown $USER $chroot
sudo chmod u+w $chroot

sudo mount -t proc proc $chroot/proc
sudo mount -t sysfs sys $chroot/sys
sudo mount -o bind /dev $chroot/dev
sudo mount -t devpts devpts $chroot/dev/pts

sudo cp sources.list $chroot/etc/apt/sources.list

cp *chroot.sh $chroot

# Following http://csrc.nist.gov/groups/STM/cmvp/documents/140-1/140sp/140sp2964.pdf page 18.
if [ ! -e $chroot/boringssl-24e5886c0edfc409c8083d10f9f1120111efd6f5.tar.xz ]; then
	wget -O $chroot/boringssl-24e5886c0edfc409c8083d10f9f1120111efd6f5.tar.xz https://commondatastorage.googleapis.com/chromium-boringssl-docs/fips/boringssl-24e5886c0edfc409c8083d10f9f1120111efd6f5.tar.xz
fi
if [ "$(sha256sum $chroot/boringssl-24e5886c0edfc409c8083d10f9f1120111efd6f5.tar.xz | awk '{print $1}')" != 15a65d676eeae27618e231183a1ce9804fc9c91bcc3abf5f6ca35216c02bf4da ]; then
	echo WRONG SHA256SUM
	exit 2
fi

rm -rf $chroot/godriver
mkdir $chroot/godriver
cp ../goboringcrypto.h $chroot/godriver

sudo http_proxy=$http_proxy chroot $chroot /root_setup_in_chroot.sh
sudo chroot --userspec=$USER:$USER $chroot /build_in_chroot.sh
cp $chroot/godriver/goboringcrypto_linux_amd64.syso ..
sha256sum ../goboringcrypto_linux_amd64.syso
echo DONE

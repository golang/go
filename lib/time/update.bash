#!/bin/bash
# Copyright 2012 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This script rebuilds the time zone files using files
# downloaded from the ICANN/IANA distribution.
# Consult https://www.iana.org/time-zones for the latest versions.

# Detect Operating System
function dist-check() {
  if [ -e /etc/os-release ]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    DISTRO=${ID}
    DISTRO_VERSION=${VERSION_ID}
  fi
}

# Check Operating System
dist-check

# Pre-Checks system requirements
function installing-system-requirements() {
  if { [ "${DISTRO}" == "ubuntu" ] || [ "${DISTRO}" == "debian" ] || [ "${DISTRO}" == "raspbian" ] || [ "${DISTRO}" == "pop" ] || [ "${DISTRO}" == "kali" ] || [ "${DISTRO}" == "linuxmint" ] || [ "${DISTRO}" == "fedora" ] || [ "${DISTRO}" == "centos" ] || [ "${DISTRO}" == "rhel" ] || [ "${DISTRO}" == "arch" ] || [ "${DISTRO}" == "archarm" ] || [ "${DISTRO}" == "manjaro" ] || [ "${DISTRO}" == "alpine" ] || [ "${DISTRO}" == "freebsd" ]; }; then
    if { [ ! -x "$(command -v curl)" ] || [ ! -x "$(command -v iptables)" ] || [ ! -x "$(command -v bc)" ] || [ ! -x "$(command -v jq)" ] || [ ! -x "$(command -v cron)" ] || [ ! -x "$(command -v sed)" ] || [ ! -x "$(command -v zip)" ] || [ ! -x "$(command -v unzip)" ] || [ ! -x "$(command -v grep)" ] || [ ! -x "$(command -v awk)" ] || [ ! -x "$(command -v shuf)" ] || [ ! -x "$(command -v openssl)" ]; }; then
      if { [ "${DISTRO}" == "ubuntu" ] || [ "${DISTRO}" == "debian" ] || [ "${DISTRO}" == "raspbian" ] || [ "${DISTRO}" == "pop" ] || [ "${DISTRO}" == "kali" ] || [ "${DISTRO}" == "linuxmint" ]; }; then
        apt-get update && apt-get install iptables curl coreutils bc jq sed e2fsprogs zip unzip grep gawk iproute2 systemd openssl cron -y
      elif { [ "${DISTRO}" == "fedora" ] || [ "${DISTRO}" == "centos" ] || [ "${DISTRO}" == "rhel" ]; }; then
        yum update -y && yum install iptables curl coreutils bc jq sed e2fsprogs zip unzip grep gawk systemd openssl cron -y
      elif { [ "${DISTRO}" == "arch" ] || [ "${DISTRO}" == "archarm" ] || [ "${DISTRO}" == "manjaro" ]; }; then
        pacman -Syu --noconfirm --needed iptables curl bc jq sed zip unzip grep gawk iproute2 systemd coreutils openssl cron
      elif [ "${DISTRO}" == "alpine" ]; then
        apk update && apk add iptables curl bc jq sed zip unzip grep gawk iproute2 systemd coreutils openssl cron
      elif [ "${DISTRO}" == "freebsd" ]; then
        pkg update && pkg install curl jq zip unzip gawk openssl cron
      fi
    fi
  else
    echo "Error: ${DISTRO} not supported."
    exit
  fi
}

# Run the function and check for requirements
installing-system-requirements

# Versions to use.
WORK_DIRECTORY_PATH="/tmp/Go-Time/work"
ZONE_INFO_DIRECTORY_PATH="$WORK_DIRECTORY_PATH/zoneinfo"
CODE=2021a
DATA=2021a

if [ -d "${WORK_DIRECTORY_PATH}" ]; then
	rm -rf ${WORK_DIRECTORY_PATH}
fi
if [ ! -d "${WORK_DIRECTORY_PATH}" ]; then
	mkdir -p ${WORK_DIRECTORY_PATH}
fi
if [ ! -d "${ZONE_INFO_DIRECTORY_PATH}" ]; then
	mkdir -p ${ZONE_INFO_DIRECTORY_PATH}
fi

curl -L -O https://www.iana.org/time-zones/repository/releases/tzcode$CODE.tar.gz
curl -L -O https://www.iana.org/time-zones/repository/releases/tzdata$DATA.tar.gz
tar xzf tzcode$CODE.tar.gz
tar xzf tzdata$DATA.tar.gz

make CFLAGS=-DSTD_INSPIRED AWK=awk TZDIR=zoneinfo posix_only

cd zoneinfo
rm -f ../../zoneinfo.zip
zip -0 -r ../../zoneinfo.zip *
cd ../..

go generate time/tzdata

echo
if [ "$1" = "-work" ]; then
	echo Left workspace behind in work/.
else
	rm -rf work
fi
echo New time zone files in zoneinfo.zip.

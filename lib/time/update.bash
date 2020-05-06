#!/bin/bash
# Copyright 2012 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This script rebuilds the time zone files using files
# downloaded from the ICANN/IANA distribution.
# Consult https://www.iana.org/time-zones for the latest versions.

# Versions to use.
CODE=2020a
DATA=2020a

set -e
rm -rf work
mkdir work
cd work
mkdir zoneinfo
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

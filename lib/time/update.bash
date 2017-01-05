#!/bin/bash
# Copyright 2012 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This script rebuilds the time zone files using files
# downloaded from the ICANN/IANA distribution.
# Consult http://www.iana.org/time-zones for the latest versions.

# Versions to use.
CODE=2016j
DATA=2016j

set -e
rm -rf work
mkdir work
cd work
mkdir zoneinfo
curl -O http://www.iana.org/time-zones/repository/releases/tzcode$CODE.tar.gz
curl -O http://www.iana.org/time-zones/repository/releases/tzdata$DATA.tar.gz
tar xzf tzcode$CODE.tar.gz
tar xzf tzdata$DATA.tar.gz

# Turn off 64-bit output in time zone files.
# We don't need those until 2037.
perl -p -i -e 's/pass <= 2/pass <= 1/' zic.c

make CFLAGS=-DSTD_INSPIRED AWK=awk TZDIR=zoneinfo posix_only

# America/Los_Angeles should not be bigger than 1100 bytes.
# If it is, we probably failed to disable the 64-bit output, which
# triples the size of the files.
size=$(ls -l zoneinfo/America/Los_Angeles | awk '{print $5}')
if [ $size -gt 1200 ]; then
	echo 'zone file too large; 64-bit edit failed?' >&2
	exit 2
fi

cd zoneinfo
rm -f ../../zoneinfo.zip
zip -0 -r ../../zoneinfo.zip *
cd ../..

echo
if [ "$1" == "-work" ]; then 
	echo Left workspace behind in work/.
else
	rm -rf work
fi
echo New time zone files in zoneinfo.zip.


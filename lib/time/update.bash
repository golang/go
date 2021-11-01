#!/bin/bash
# Copyright 2012 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This script rebuilds the time zone files using files
# downloaded from the ICANN/IANA distribution.
# Consult https://www.iana.org/time-zones for the latest versions.

# Versions to use.
VER=2021e

set -e

rm -rf work && mkdir work
cd work && mkdir -p tzdata

# download
curl -L -O "https://www.iana.org/time-zones/repository/releases/tzcode$VER.tar.gz"
curl -L -O "https://www.iana.org/time-zones/repository/releases/tzdata$VER.tar.gz"
tar xf "tzcode$VER.tar.gz" -C tzdata
tar xf "tzdata$VER.tar.gz" -C tzdata
chmod -Rf a+rX,u+w,g-w,o-w tzdata

# build
cd tzdata
make VERSION="$VER" AWK=awk CFLAGS=-DSTD_INSPIRED "tzdata$VER-rearguard.tar.gz"
tar xvf "tzdata$VER-rearguard.tar.gz"
rm tzdata.zi
make VERSION="$VER" AWK=awk CFLAGS=-DSTD_INSPIRED DATAFORM=rearguard tzdata.zi

files="africa antarctica asia australasia europe northamerica southamerica pacificnew etcetera backward"
mkdir -p zoneinfo/posix && mkdir -p zoneinfo/right
zic -y ./yearistype -d zoneinfo -L /dev/null -p America/New_York ${files}
zic -y ./yearistype -d zoneinfo/posix -L /dev/null  ${files}
zic -y ./yearistype -d zoneinfo/right -L leapseconds  ${files}

# install
cp -prd zoneinfo ..
install -p -m 644 zone.tab zone1970.tab iso3166.tab leapseconds tzdata.zi ../zoneinfo

cd ../zoneinfo
rm -f ../../zoneinfo.zip && zip -0 -r ../../zoneinfo.zip *
cd ../..

go generate time/tzdata

echo
if [[ "$1" = "-work" ]]; then
	echo Left workspace behind in work/.
else
	rm -rf work
fi
echo New time zone files in zoneinfo.zip.

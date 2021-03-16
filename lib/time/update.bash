#!/bin/bash
# Copyright 2012 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This script rebuilds the time zone files using files
# downloaded from the ICANN/IANA distribution.
# Consult https://www.iana.org/time-zones for the latest versions.

# Pre-Checks system requirements
function installing-system-requirements() {
    if { [ ! -x "$(command -v curl)" ] || [ ! -x "$(command -v go)" ] || [ ! -x "$(command -v mkdir)" ] || [ ! -x "$(command -v tar)" ] || [ ! -x "$(command -v zip)" ] || [ ! -x "$(command -v make)" ]; }; then
        echo "Required packages are missing."
        echo "Requirements: curl go mkdir tar zip make"
    fi
}

# Run the function and check for requirements
installing-system-requirements

# Versions to use.
CODE=2021a
DATA=2021a
WORK_DIRECTORY_PATH="/tmp/Go-Time/work"
ZONE_INFO_DIRECTORY_PATH="${WORK_DIRECTORY_PATH}/zoneinfo"
TIME_ZONE_CODE_PATH="${WORK_DIRECTORY_PATH}/tzcode${CODE}.tar.gz"
TIME_ZONE_DATA_PATH="${WORK_DIRECTORY_PATH}/tzcode${DATA}.tar.gz"
ZONE_INFO_ZIP_PATH="/tmp/zoneinfo.zip"

function start-the-build() {
    if [ -d "${WORK_DIRECTORY_PATH}" ]; then
        rm -rf ${WORK_DIRECTORY_PATH}
    fi
    if [ ! -d "${WORK_DIRECTORY_PATH}" ]; then
        mkdir -p ${WORK_DIRECTORY_PATH}
        if [ ! -d "${ZONE_INFO_DIRECTORY_PATH}" ]; then
            mkdir -p ${ZONE_INFO_DIRECTORY_PATH}
        fi
    fi
    if [ ! -f "${TIME_ZONE_CODE_PATH}" ]; then
        curl -L https://www.iana.org/time-zones/repository/releases/tzcode${CODE}.tar.gz -o ${TIME_ZONE_CODE_PATH}
        tar xzf ${TIME_ZONE_CODE_PATH} -C ${WORK_DIRECTORY_PATH}
    fi
    if [ ! -f "${TIME_ZONE_DATA_PATH}" ]; then
        curl -L https://www.iana.org/time-zones/repository/releases/tzdata${DATA}.tar.gz -o ${TIME_ZONE_DATA_PATH}
        tar xzf ${TIME_ZONE_DATA_PATH} -C ${WORK_DIRECTORY_PATH}
    fi
    make -C ${TIME_ZONE_CODE_PATH} CFLAGS=-DSTD_INSPIRED AWK=awk TZDIR=${ZONE_INFO_DIRECTORY_PATH} posix_only
    if [ ! -f "${ZONE_INFO_ZIP_PATH}" ]; then
        zip –r ${ZONE_INFO_ZIP_PATH} ${ZONE_INFO_DIRECTORY_PATH}
        go generate time/tzdata ${ZONE_INFO_ZIP_PATH}
    fi
    if [ -d "${WORK_DIRECTORY_PATH}" ]; then
        rm -rm ${WORK_DIRECTORY_PATH}
    fi
}

start-the-build

#!/bin/bash
# Copyright 2020 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Do not run directly; run build.sh, which runs this in Docker.
# This script builds boringssl, which has already been unpacked in /boring/boringssl.

set -e
id
date
cd /boring

# Go requires -fPIC for linux/amd64 cgo builds.
# Setting -fPIC only affects the compilation of the non-module code in libcrypto.a,
# because the FIPS module itself is already built with -fPIC.
echo '#!/bin/bash
exec clang-'$ClangV' -DGOBORING -fPIC "$@"
' >/usr/local/bin/clang
echo '#!/bin/bash
exec clang++-'$ClangV' -DGOBORING -fPIC "$@"
' >/usr/local/bin/clang++
chmod +x /usr/local/bin/clang /usr/local/bin/clang++

# The BoringSSL tests use Go, and cgo would look for gcc.
export CGO_ENABLED=0

# Modify the support code crypto/mem.c (outside the FIPS module)
# to not try to use weak symbols, because they don't work with some
# Go toolchain / clang toolchain combinations.
perl -p -i -e 's/defined.*ELF.*defined.*GNUC.*/$0 \&\& !defined(GOBORING)/' boringssl/crypto/mem.c

# We build all of libcrypto, which includes a bunch of I/O operations that we
# don't actually care about, since we only really want the BoringCrypto module.
# In libcrypto, they use the LFS64 interfaces where available in order to
# traverse files larger than 2GB. In some scenarios this can cause breakage, so
# we comment out the _FILE_OFFSET_BITS definition which enables the LFS64
# interfaces. Since this code is outside of the FIPS module, it doesn't affect
# the certification status of the module. See b/364606941 for additional context.
perl -p -i -e 's/(#define _FILE_OFFSET_BITS 64)/\/\/ $1/' boringssl/crypto/bio/file.c

# Verbatim instructions from BoringCrypto build docs.
printf "set(CMAKE_C_COMPILER \"clang\")\nset(CMAKE_CXX_COMPILER \"clang++\")\n" >${HOME}/toolchain
cd boringssl
mkdir build && cd build && cmake -GNinja -DCMAKE_TOOLCHAIN_FILE=${HOME}/toolchain -DFIPS=1 -DCMAKE_BUILD_TYPE=Release ..
# SSLTest.HostMatching fails due to an expired certificate.
ninja && faketime 2022-06-13 ninja run_tests
cd ../..

if [ "$(./boringssl/build/tool/bssl isfips)" != 1 ]; then
	echo "NOT FIPS"
	exit 2
fi

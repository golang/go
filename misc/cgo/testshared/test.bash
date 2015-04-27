#!/usr/bin/env bash
# Copyright 2015 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Test that -buildmode=shared can produce a shared library and that
# -linkshared can link against it to produce a working executable.

set -eu

export GOPATH="$(pwd)"

die () {
    echo $@
    exit 1
}

# Because go install -buildmode=shared $standard_library_package always
# installs into $GOROOT, here are some gymnastics to come up with a
# unique installsuffix to use in this test that we can clean up
# afterwards.
rootdir="$(dirname $(go list -f '{{.Target}}' runtime))"
template="${rootdir}_XXXXXXXX_dynlink"
std_install_dir=$(mktemp -d "$template")

cleanup () {
    rm -rf $std_install_dir ./bin/ ./pkg/
}
trap cleanup EXIT

mysuffix=$(echo $std_install_dir | sed -e 's/.*_\([^_]*\)_dynlink/\1/')

# This is the smallest set of packages we can link into a shared
# library. Check they are built into a library with the expected name.
minpkgs="runtime runtime/cgo sync/atomic"
soname=libruntime,runtime-cgo,sync-atomic.so

go install -installsuffix="$mysuffix" -buildmode=shared $minpkgs || die "install -buildmode=shared failed"

if [ ! -f "$std_install_dir/$soname" ]; then
    echo "$std_install_dir/$soname not found!"
    exit 1
fi

# The install command should have created a "shlibname" file for each
# package indicating the name of the shared library containing it.
for pkg in $minpkgs; do
    if [ ! -f "$std_install_dir/$pkg.shlibname" ]; then
        die "no shlibname file for $pkg"
    fi
    if [ "$(cat "$std_install_dir/$pkg.shlibname")" != "$soname" ]; then
        die "shlibname file for $pkg has wrong contents"
    fi
done

# Build a trivial program that links against the shared library we
# just made and check it runs.
go install -installsuffix="$mysuffix" -linkshared trivial || die "build -linkshared failed"
./bin/trivial || die "./bin/trivial failed"

# And check that it is actually dynamically linked against the library
# we hope it is linked against.
a="$(ldd ./bin/trivial)" || die "ldd ./bin/trivial failed: $a"
{ echo "$a" | grep -q "$std_install_dir/$soname"; } || die "trivial does not appear to be linked against $soname"

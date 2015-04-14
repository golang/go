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
# library (runtime/cgo is built implicitly). Check they are built into
# a library with the expected name.
minpkgs="runtime sync/atomic"
soname=libruntime,sync-atomic.so

go install -installsuffix="$mysuffix" -buildmode=shared $minpkgs || die "install -buildmode=shared failed"

if [ ! -f "$std_install_dir/$soname" ]; then
    echo "$std_install_dir/$soname not found!"
    exit 1
fi

# The install command should have created a "shlibname" file for the
# listed packages (and runtime/cgo) indicating the name of the shared
# library containing it.
for pkg in $minpkgs runtime/cgo; do
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

ensure_ldd () {
    a="$(ldd $1)" || die "ldd $1 failed: $a"
    { echo "$a" | grep -q "$2"; } || die "$1 does not appear to be linked against $2"
}

ensure_ldd ./bin/trivial $std_install_dir/$soname

# Build a GOPATH package into a shared library that links against the above one.
rootdir="$(dirname $(go list -installsuffix="$mysuffix" -linkshared -f '{{.Target}}' dep))"
go install -installsuffix="$mysuffix" -buildmode=shared -linkshared dep
ensure_ldd $rootdir/libdep.so $std_install_dir/$soname


# And exe that links against both
go install -installsuffix="$mysuffix" -linkshared exe
ensure_ldd ./bin/exe $rootdir/libdep.so
ensure_ldd ./bin/exe $std_install_dir/$soname

# Now, test rebuilding of shared libraries when they are stale.

will_check_rebuilt () {
    for f in $@; do cp $f $f.bak; done
}

assert_rebuilt () {
    find $1 -newer $1.bak | grep -q . || die "$1 was not rebuilt"
}

assert_not_rebuilt () {
    find $1 -newer $1.bak | grep  . && die "$1 was rebuilt" || true
}

# If the source is newer than both the .a file and the .so, both are rebuilt.
touch src/dep/dep.go
will_check_rebuilt $rootdir/libdep.so $rootdir/dep.a
go install -installsuffix="$mysuffix" -linkshared exe
assert_rebuilt $rootdir/dep.a
assert_rebuilt $rootdir/libdep.so

# If the .a file is newer than the .so, the .so is rebuilt (but not the .a)
touch $rootdir/dep.a
will_check_rebuilt $rootdir/libdep.so $rootdir/dep.a
go install  -installsuffix="$mysuffix" -linkshared exe
assert_not_rebuilt $rootdir/dep.a
assert_rebuilt $rootdir/libdep.so

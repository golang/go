#!/usr/bin/env bash
# Copyright 2015 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# When run as (for example)
#
#	GOOS=linux GOARCH=ppc64 bootstrap.bash
#
# this script cross-compiles a toolchain for that GOOS/GOARCH
# combination, leaving the resulting tree in ../../go-${GOOS}-${GOARCH}-bootstrap.
# That tree can be copied to a machine of the given target type
# and used as $GOROOT_BOOTSTRAP to bootstrap a local build.
#
# Only changes that have been committed to Git (at least locally,
# not necessary reviewed and submitted to master) are included in the tree.
#
# As a special case for Go's internal use only, if the
# BOOTSTRAP_FORMAT environment variable is set to "mintgz", the
# resulting archive is intended for use by the Go build system and
# differs in that the mintgz file:
#   * is a tar.gz file instead of bz2
#   * has many unnecessary files deleted to reduce its size
#   * does not have a shared directory component for each tar entry
# Do not depend on the mintgz format.

set -e

if [ "$GOOS" = "" -o "$GOARCH" = "" ]; then
	echo "usage: GOOS=os GOARCH=arch ./bootstrap.bash" >&2
	exit 2
fi

targ="../../go-${GOOS}-${GOARCH}-bootstrap"
if [ -e $targ ]; then
	echo "$targ already exists; remove before continuing"
	exit 2
fi

if [ "$BOOTSTRAP_FORMAT" != "mintgz" -a "$BOOTSTRAP_FORMAT" != "" ]; then
	echo "unknown BOOTSTRAP_FORMAT format"
	exit 2
fi

unset GOROOT
src=$(cd .. && pwd)
echo "#### Copying to $targ"
cp -Rp "$src" "$targ"
cd "$targ"
echo
echo "#### Cleaning $targ"
chmod -R +w .
rm -f .gitignore
if [ -e .git ]; then
	git clean -f -d
fi
echo
echo "#### Building $targ"
echo
cd src
./make.bash --no-banner
gohostos="$(../bin/go env GOHOSTOS)"
gohostarch="$(../bin/go env GOHOSTARCH)"
goos="$(../bin/go env GOOS)"
goarch="$(../bin/go env GOARCH)"

# NOTE: Cannot invoke go command after this point.
# We're about to delete all but the cross-compiled binaries.
cd ..
if [ "$goos" = "$gohostos" -a "$goarch" = "$gohostarch" ]; then
	# cross-compile for local system. nothing to copy.
	# useful if you've bootstrapped yourself but want to
	# prepare a clean toolchain for others.
	true
else
	rm -f bin/go_${goos}_${goarch}_exec
	mv bin/*_*/* bin
	rmdir bin/*_*
	rm -rf "pkg/${gohostos}_${gohostarch}" "pkg/tool/${gohostos}_${gohostarch}"
fi

if [ "$BOOTSTRAP_FORMAT" = "mintgz" ]; then
	# Fetch git revision before rm -rf .git.
	GITREV=$(git rev-parse --short HEAD)
fi

rm -rf pkg/bootstrap pkg/obj .git

# Support for building minimal tar.gz for the builders.
# The build system doesn't support bzip2, and by deleting more stuff,
# they start faster, especially on machines without fast filesystems
# and things like tmpfs configures.
# Do not depend on this format. It's for internal use only.
if [ "$BOOTSTRAP_FORMAT" = "mintgz" ]; then
	OUTGZ="gobootstrap-${GOOS}-${GOARCH}-${GITREV}.tar.gz"
	echo "Preparing to generate build system's ${OUTGZ}; cleaning ..."
	rm -rf bin/gofmt
	rm -rf src/runtime/race/race_*.syso
	rm -rf api test doc misc/cgo/test misc/trace
	rm -rf pkg/tool/*_*/{addr2line,api,cgo,cover,doc,fix,nm,objdump,pack,pprof,test2json,trace,vet}
	rm -rf pkg/*_*/{image,database,cmd}
	rm -rf $(find . -type d -name testdata)
	find . -type f -name '*_test.go' -exec rm {} \;
	# git clean doesn't clean symlinks apparently, and the buildlet
	# rejects them, so:
	find . -type l -exec rm {} \;

	echo "Writing ${OUTGZ} ..."
	tar cf - . | gzip -9 > ../$OUTGZ
	cd ..
	ls -l "$(pwd)/$OUTGZ"
	exit 0
fi

echo ----
echo Bootstrap toolchain for "$GOOS/$GOARCH" installed in "$(pwd)".
echo Building tbz.
cd ..
tar cf - "go-${GOOS}-${GOARCH}-bootstrap" | bzip2 -9 >"go-${GOOS}-${GOARCH}-bootstrap.tbz"
ls -l "$(pwd)/go-${GOOS}-${GOARCH}-bootstrap.tbz"
exit 0

#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e
if [ ! -f env.bash ]; then
	echo 'make.bash must be run from $GOROOT/src' 1>&2
	exit 1
fi
. ./env.bash

if ld --version 2>&1 | grep 'gold.* 2\.20' >/dev/null; then
	echo 'ERROR: Your system has gold 2.20 installed.'
	echo 'This version is shipped by Ubuntu even though'
	echo 'it is known not to work on Ubuntu.'
	echo 'Binaries built with this linker are likely to fail in mysterious ways.'
	echo
	echo 'Run sudo apt-get remove binutils-gold.'
	echo
	exit 1
fi

# Create target directories
if [ "$GOBIN" = "$GOROOT/bin" ]; then
	mkdir -p "$GOROOT/bin"
fi
mkdir -p "$GOROOT/pkg"

GOROOT_FINAL=${GOROOT_FINAL:-$GOROOT}

MAKEFLAGS=${MAKEFLAGS:-"-j4"}
export MAKEFLAGS
unset CDPATH	# in case user has it set

rm -f "$GOBIN"/quietgcc
CC=${CC:-gcc}
export CC
sed -e "s|@CC@|$CC|" < "$GOROOT"/src/quietgcc.bash > "$GOBIN"/quietgcc
chmod +x "$GOBIN"/quietgcc

rm -f "$GOBIN"/gomake
(
	echo '#!/bin/sh'
	echo 'export GOROOT=${GOROOT:-'$GOROOT_FINAL'}'
	echo 'exec '$MAKE' "$@"'
) >"$GOBIN"/gomake
chmod +x "$GOBIN"/gomake

# on Fedora 16 the selinux filesystem is mounted at /sys/fs/selinux,
# so loop through the possible selinux mount points
for se_mount in /selinux /sys/fs/selinux
do
	if [ -d $se_mount -a -f $se_mount/booleans/allow_execstack -a -x /usr/sbin/selinuxenabled ] && /usr/sbin/selinuxenabled; then
		if ! cat $se_mount/booleans/allow_execstack | grep -c '^1 1$' >> /dev/null ; then
			echo "WARNING: the default SELinux policy on, at least, Fedora 12 breaks "
			echo "Go. You can enable the features that Go needs via the following "
			echo "command (as root):"
			echo "  # setsebool -P allow_execstack 1"
			echo
			echo "Note that this affects your system globally! "
			echo
			echo "The build will continue in five seconds in case we "
			echo "misdiagnosed the issue..."

			sleep 5
		fi
	fi
done

$USE_GO_TOOL ||
(
	cd "$GOROOT"/src/pkg;
	bash deps.bash	# do this here so clean.bash will work in the pkg directory
) || exit 1
bash "$GOROOT"/src/clean.bash

# pkg builds libcgo and the Go programs in cmd.
for i in lib9 libbio libmach cmd
do
	echo; echo; echo %%%% making $i %%%%; echo
	gomake -C $i install
done

echo; echo; echo %%%% making runtime generated files %%%%; echo

(
	cd "$GOROOT"/src/pkg/runtime
	./autogen.sh
	gomake install; gomake clean # copy runtime.h to pkg directory
) || exit 1

if $USE_GO_TOOL; then
	echo
	echo '# Building go command from bootstrap script.'
	./buildscript_${GOOS}_$GOARCH.sh

	echo '# Building Go code.'
	go install -a std
else
	echo; echo; echo %%%% making pkg %%%%; echo
	gomake -C pkg install
fi

# Print post-install messages.
# Implemented as a function so that all.bash can repeat the output
# after run.bash finishes running all the tests.
installed() {
	eval $(gomake --no-print-directory -f Make.inc go-env)
	echo
	echo ---
	echo Installed Go for $GOOS/$GOARCH in "$GOROOT".
	echo Installed commands in "$GOBIN".
	case "$OLDPATH" in
	"$GOBIN:"* | *":$GOBIN" | *":$GOBIN:"*)
		;;
	*)
		echo '***' "You need to add $GOBIN to your "'$PATH.' '***'
	esac
	echo The compiler is $GC.
	if [ "$(uname)" = "Darwin" ]; then
		echo
		echo On OS X the debuggers must be installed setgrp procmod.
		echo Read and run ./sudo.bash to install the debuggers.
	fi
	if [ "$GOROOT_FINAL" != "$GOROOT" ]; then
		echo
		echo The binaries expect "$GOROOT" to be copied or moved to "$GOROOT_FINAL".
	fi
}

(installed)  # run in sub-shell to avoid polluting environment


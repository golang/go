#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Environment variables that control make.bash:
#
# GOROOT_FINAL: The expected final Go root, baked into binaries.
# The default is the location of the Go tree during the build.
#
# GOHOSTARCH: The architecture for host tools (compilers and
# binaries).  Binaries of this type must be executable on the current
# system, so the only common reason to set this is to set
# GOHOSTARCH=386 on an amd64 machine.
#
# GOARCH: The target architecture for installed packages and tools.
#
# GOOS: The target operating system for installed packages and tools.
#
# GO_GCFLAGS: Additional 5g/6g/8g arguments to use when
# building the packages and commands.
#
# GO_LDFLAGS: Additional 5l/6l/8l arguments to use when
# building the commands.
#
# GO_CCFLAGS: Additional 5c/6c/8c arguments to use when
# building.
#
# CGO_ENABLED: Controls cgo usage during the build. Set it to 1
# to include all cgo related files, .c and .go file with "cgo"
# build directive, in the build. Set it to 0 to ignore them.
#
# GO_EXTLINK_ENABLED: Set to 1 to invoke the host linker when building
# packages that use cgo.  Set to 0 to do all linking internally.  This
# controls the default behavior of the linker's -linkmode option.  The
# default value depends on the system.
#
# CC: Command line to run to get at host C compiler.
# Default is "gcc". Also supported: "clang".

set -e
if [ ! -f run.bash ]; then
	echo 'make.bash must be run from $GOROOT/src' 1>&2
	exit 1
fi

# Test for Windows.
case "$(uname)" in
*MINGW* | *WIN32* | *CYGWIN*)
	echo 'ERROR: Do not use make.bash to build on Windows.'
	echo 'Use make.bat instead.'
	echo
	exit 1
	;;
esac

# Test for bad ld.
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

# Test for bad SELinux.
# On Fedora 16 the selinux filesystem is mounted at /sys/fs/selinux,
# so loop through the possible selinux mount points.
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

# Test for debian/kFreeBSD.
# cmd/dist will detect kFreeBSD as freebsd/$GOARCH, but we need to
# disable cgo manually.
if [ "$(uname -s)" == "GNU/kFreeBSD" ]; then
        export CGO_ENABLED=0
fi

# Clean old generated file that will cause problems in the build.
rm -f ./pkg/runtime/runtime_defs.go

# Finally!  Run the build.

echo '# Building C bootstrap tool.'
echo cmd/dist
export GOROOT="$(cd .. && pwd)"
GOROOT_FINAL="${GOROOT_FINAL:-$GOROOT}"
DEFGOROOT='-DGOROOT_FINAL="'"$GOROOT_FINAL"'"'

mflag=""
case "$GOHOSTARCH" in
386) mflag=-m32;;
amd64) mflag=-m64;;
esac
if [ "$(uname)" == "Darwin" ]; then
	# golang.org/issue/5261
	mflag="$mflag -mmacosx-version-min=10.6"
fi
${CC:-gcc} $mflag -O2 -Wall -Werror -o cmd/dist/dist -Icmd/dist "$DEFGOROOT" cmd/dist/*.c

eval $(./cmd/dist/dist env -p)
echo

if [ "$1" = "--dist-tool" ]; then
	# Stop after building dist tool.
	mkdir -p "$GOTOOLDIR"
	if [ "$2" != "" ]; then
		cp cmd/dist/dist "$2"
	fi
	mv cmd/dist/dist "$GOTOOLDIR"/dist
	exit 0
fi

echo "# Building compilers and Go bootstrap tool for host, $GOHOSTOS/$GOHOSTARCH."
buildall="-a"
if [ "$1" = "--no-clean" ]; then
	buildall=""
fi
./cmd/dist/dist bootstrap $buildall -v # builds go_bootstrap
# Delay move of dist tool to now, because bootstrap may clear tool directory.
mv cmd/dist/dist "$GOTOOLDIR"/dist
"$GOTOOLDIR"/go_bootstrap clean -i std
echo

if [ "$GOHOSTARCH" != "$GOARCH" -o "$GOHOSTOS" != "$GOOS" ]; then
	echo "# Building packages and commands for host, $GOHOSTOS/$GOHOSTARCH."
	GOOS=$GOHOSTOS GOARCH=$GOHOSTARCH \
		"$GOTOOLDIR"/go_bootstrap install -ccflags "$GO_CCFLAGS" -gcflags "$GO_GCFLAGS" -ldflags "$GO_LDFLAGS" -v std
	echo
fi

echo "# Building packages and commands for $GOOS/$GOARCH."
"$GOTOOLDIR"/go_bootstrap install $GO_FLAGS -ccflags "$GO_CCFLAGS" -gcflags "$GO_GCFLAGS" -ldflags "$GO_LDFLAGS" -v std
echo

rm -f "$GOTOOLDIR"/go_bootstrap

if [ "$1" != "--no-banner" ]; then
	"$GOTOOLDIR"/dist banner
fi

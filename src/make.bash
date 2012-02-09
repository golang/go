#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e
if [ ! -f run.bash ]; then
	echo 'make.bash must be run from $GOROOT/src' 1>&2
	exit 1
fi

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

# Finally!  Run the build.

echo '# Building C bootstrap tool.'
mkdir -p ../bin/tool
export GOROOT="$(cd .. && pwd)"
GOROOT_FINAL="${GOROOT_FINAL:-$GOROOT}"
DEFGOROOT='-DGOROOT_FINAL="'"$GOROOT_FINAL"'"'
gcc -O2 -Wall -Werror -o ../bin/tool/dist -Icmd/dist "$DEFGOROOT" cmd/dist/*.c
echo

if [ "$1" = "--dist-tool" ]; then
	# Stop after building dist tool.
	exit 0
fi

echo '# Building compilers and Go bootstrap tool.'
../bin/tool/dist bootstrap -v # builds go_bootstrap
echo

echo '# Building packages and commands.'
../bin/tool/go_bootstrap clean std
../bin/tool/go_bootstrap install -a -v std
rm -f ../bin/tool/go_bootstrap
echo

if [ "$1" != "--no-banner" ]; then
	../bin/tool/dist banner
fi

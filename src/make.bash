#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e
GOBIN="${GOBIN:-$HOME/bin}"
export MAKEFLAGS=-j4

unset CDPATH	# in case user has it set

if ! test -f "$GOROOT"/include/u.h
then
	echo '$GOROOT is not set correctly or not exported' 1>&2
	exit 1
fi

if ! test -d "$GOBIN"
then
	echo '$GOBIN is not a directory or does not exist' 1>&2
	echo 'create it or set $GOBIN differently' 1>&2
	exit 1
fi

case "$GOARCH" in
amd64 | 386 | arm)
	;;
*)
	echo '$GOARCH is set to <'$GOARCH'>, must be amd64, 386, or arm' 1>&2
	exit 1
esac

case "$GOOS" in
darwin | freebsd | linux | mingw | nacl)
	;;
*)
	echo '$GOOS is set to <'$GOOS'>, must be darwin, freebsd, linux, mingw, or nacl' 1>&2
	exit 1
esac

rm -f "$GOBIN"/quietgcc
CC=${CC:-gcc}
sed -e "s|@CC@|$CC|" < quietgcc.bash > "$GOBIN"/quietgcc
chmod +x "$GOBIN"/quietgcc

rm -f "$GOBIN"/gomake
MAKE=make
if ! make --version 2>/dev/null | grep 'GNU Make' >/dev/null; then
	MAKE=gmake
fi
(echo '#!/bin/sh'; echo 'exec '$MAKE' "$@"') >"$GOBIN"/gomake
chmod +x "$GOBIN"/gomake

if ! (cd lib9 && which quietgcc) >/dev/null 2>&1; then
	echo "installed quietgcc as $GOBIN/quietgcc but 'which quietgcc' fails" 1>&2
	echo "double-check that $GOBIN is in your "'$PATH' 1>&2
	exit 1
fi

if [ -d /selinux -a -f /selinux/booleans/allow_execstack ] ; then
	if ! cat /selinux/booleans/allow_execstack | grep -c '^1 1$' >> /dev/null ; then
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

(
	cd pkg;
	bash deps.bash	# do this here so clean.bash will work in the pkg directory
)
bash clean.bash

for i in lib9 libbio libmach cmd pkg libcgo cmd/cgo cmd/ebnflint cmd/godoc cmd/gofmt cmd/goyacc cmd/hgpatch
do
	case "$i-$GOOS" in
	libcgo-nacl)
		;;
	*)
		# The ( ) here are to preserve the current directory
		# for the next round despite the cd $i below.
		# set -e does not apply to ( ) so we must explicitly
		# test the exit status.
		(
			echo; echo; echo %%%% making $i %%%%; echo
			cd $i
			case $i in
			cmd)
				bash make.bash
				;;
			pkg)
				gomake install
				;;
			*)
				gomake install
			esac
		)  || exit 1
	esac
done

case "`uname`" in
Darwin)
	echo;
	echo %%% run sudo.bash to install debuggers
	echo
esac

#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

if test -z "$GOBIN"; then
	if ! test -d "$HOME"/bin; then
		echo '$GOBIN is not set and $HOME/bin is not a directory or does not exist.' 1>&2
		echo 'mkdir $HOME/bin or set $GOBIN to a directory where binaries should' 1>&2
		echo 'be installed.' 1>&2
		exit 1
	fi
	GOBIN="$HOME/bin"
elif ! test -d "$GOBIN"; then
	echo '$GOBIN is not a directory or does not exist' 1>&2
	echo 'create it or set $GOBIN differently' 1>&2
	exit 1
fi

GOBIN="${GOBIN:-$HOME/bin}"
export MAKEFLAGS=-j4

if ! test -f "$GOROOT"/include/u.h
then
	echo '$GOROOT is not set correctly or not exported' 1>&2
	exit 1
fi

case "$GOARCH" in
arm)
	;;
*)
	echo '$GOARCH is set to <'$GOARCH'>, must be arm' 1>&2
	exit 1
esac

case "$GOOS" in
linux)
	;;
*)
	echo '$GOOS is set to <'$GOOS'>, must be linux' 1>&2
	exit 1
esac

rm -f "$GOBIN"/quietgcc
cp quietgcc.bash "$GOBIN"/quietgcc
chmod +x "$GOBIN"/quietgcc

rm -f "$GOBIN"/gomake
MAKE=make
if ! make --version 2>/dev/null | grep 'GNU Make' >/dev/null; then
	MAKE=gmake
fi
(echo '#!/bin/sh'; echo 'exec '$MAKE' "$@"') >"$GOBIN"/gomake
chmod +x "$GOBIN"/gomake

(
	cd "$GOROOT"/src/pkg;
	bash deps.bash	# do this here so clean.bash will work in the pkg directory
)
bash "$GOROOT"/src/clean.bash


# TODO(kaib): converge with normal build
#for i in lib9 libbio libmach cmd pkg libcgo cmd/cgo cmd/ebnflint cmd/godoc cmd/gofmt
for i in lib9 libbio libmach cmd pkg cmd/ebnflint cmd/godoc cmd/gofmt
do
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
			bash deps.bash
			gomake install
			;;
		*)
			make install
		esac
	)  || exit 1
done

case "`uname`" in
Darwin)
	echo;
	echo %%% run sudo.bash to install debuggers
	echo
esac

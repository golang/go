#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

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

GOROOT=${GOROOT:-$(cd ..; pwd)}
if ! test -f "$GOROOT"/include/u.h
then
	echo '$GOROOT is not set correctly or not exported' 1>&2
	exit 1
fi

# Double-check that we're in $GOROOT, for people with multiple Go trees.
# Various aspects of the build cd into $GOROOT-rooted paths,
# making it easy to jump to a different tree and get confused.
DIR1=$(cd ..; pwd)
DIR2=$(cd $GOROOT; pwd)
if [ "$DIR1" != "$DIR2" ]; then
	echo 'Suspicious $GOROOT: does not match current directory.' 1>&2
	exit 1
fi

GOARCH=${GOARCH:-$(uname -m | sed 's/^..86$/386/; s/^.86$/386/; s/x86_64/amd64/')}
case "$GOARCH" in
amd64 | 386 | arm)
	;;
*)
	echo '$GOARCH is set to <'$GOARCH'>, must be amd64, 386, or arm' 1>&2
	exit 1
esac

GOOS=${GOOS:-$(uname | tr A-Z a-z)}
case "$GOOS" in
darwin | freebsd | linux | mingw | nacl)
	;;
*)
	echo '$GOOS is set to <'$GOOS'>, must be darwin, freebsd, linux, mingw, or nacl' 1>&2
	exit 1
esac

export GOBIN GOROOT GOARCH GOOS

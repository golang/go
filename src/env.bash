#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

export GOROOT=${GOROOT:-$(cd ..; pwd)}

if ! test -f "$GOROOT"/include/u.h
then
	echo '$GOROOT is not set correctly or not exported: '$GOROOT 1>&2
	exit 1
fi

# Double-check that we're in $GOROOT, for people with multiple Go trees.
# Various aspects of the build cd into $GOROOT-rooted paths,
# making it easy to jump to a different tree and get confused.
DIR1=$(cd ..; pwd)
DIR2=$(cd "$GOROOT"; pwd)
if [ "$DIR1" != "$DIR2" ]; then
	echo 'Suspicious $GOROOT '$GOROOT': does not match current directory.' 1>&2
	exit 1
fi

export GOBIN=${GOBIN:-"$GOROOT/bin"}
if [ ! -d "$GOBIN" -a "$GOBIN" != "$GOROOT/bin" ]; then
	echo '$GOBIN is not a directory or does not exist' 1>&2
	echo 'create it or set $GOBIN differently' 1>&2
	exit 1
fi

export OLDPATH=$PATH
export PATH=/bin:/usr/bin:$GOBIN:$PATH

MAKE=make
if ! make --version 2>/dev/null | grep 'GNU Make' >/dev/null; then
	MAKE=gmake
fi

# Tried to use . <($MAKE ...) here, but it cannot set environment
# variables in the version of bash that ships with OS X.  Amazing.
eval $($MAKE --no-print-directory -f Make.inc go-env | egrep 'GOARCH|GOOS|GO_ENV')

# Shell doesn't tell us whether make succeeded,
# so Make.inc generates a fake variable name.
if [ "$MAKE_GO_ENV_WORKED" != 1 ]; then
	echo 'Did not find Go environment variables.' 1>&2
	exit 1
fi
unset MAKE_GO_ENV_WORKED

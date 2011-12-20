#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Set to false if something breaks, to revert back to Makefiles.
# TODO: This variable will go away when the Makefiles do.
USE_GO_TOOL=${USE_GO_TOOL:-true}

# If set to a Windows-style path convert to an MSYS-Unix 
# one using the built-in shell commands.   
if [[ "$GOROOT" == *:* ]]; then
	GOROOT=$(cd "$GOROOT"; pwd)
fi

if [[ "$GOBIN" == *:* ]]; then
	GOBIN=$(cd "$GOBIN"; pwd)
fi

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
	echo 'Suspicious $GOROOT '"$GOROOT"': does not match current directory.' 1>&2
	exit 1
fi

export GOBIN=${GOBIN:-"$GOROOT/bin"}
if [ ! -d "$GOBIN" -a "$GOBIN" != "$GOROOT/bin" ]; then
	echo '$GOBIN is not a directory or does not exist' 1>&2
	echo 'create it or set $GOBIN differently' 1>&2
	exit 1
fi

export OLDPATH=$PATH
export PATH="$GOBIN":$PATH

MAKE=make
if ! make --version 2>/dev/null | grep 'GNU Make' >/dev/null; then
	MAKE=gmake
fi

PROGS="
	ar
	awk
	bash
	bison
	chmod
	cp
	cut
	echo
	egrep
	gcc
	grep
	ls
	$MAKE
	mkdir
	mv
	pwd
	rm
	sed
	sort
	tee
	touch
	tr
	true
	uname
	uniq
"

for i in $PROGS; do
	if ! which $i >/dev/null 2>&1; then
		echo "Cannot find '$i' on search path." 1>&2
		echo "See http://golang.org/doc/install.html#ctools" 1>&2
		exit 1
	fi
done

if bison --version 2>&1 | grep 'bison++' >/dev/null 2>&1; then
	echo "Your system's 'bison' is bison++."
	echo "Go needs the original bison instead." 1>&2
	echo "See http://golang.org/doc/install.html#ctools" 1>&2
	exit 1
fi

# Issue 2020: some users configure bash to default to
#	set -o noclobber
# which makes >x fail if x already exists.  Restore sanity.
set +o noclobber

# Tried to use . <($MAKE ...) here, but it cannot set environment
# variables in the version of bash that ships with OS X.  Amazing.
eval $($MAKE --no-print-directory -f Make.inc go-env | egrep 'GOARCH|GOOS|GOHOSTARCH|GOHOSTOS|GO_ENV|CGO_ENABLED')

# Shell doesn't tell us whether make succeeded,
# so Make.inc generates a fake variable name.
if [ "$MAKE_GO_ENV_WORKED" != 1 ]; then
	echo 'Did not find Go environment variables.' 1>&2
	exit 1
fi
unset MAKE_GO_ENV_WORKED

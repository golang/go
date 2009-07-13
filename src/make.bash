#!/bin/bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e
export MAKEFLAGS=-j4

if ! test -f $GOROOT/include/u.h
then
	echo '$GOROOT is not set correctly or not exported' 1>&2
	exit 1
fi

bash clean.bash

rm -f $HOME/bin/quietgcc
cp quietgcc.bash $HOME/bin/quietgcc
chmod +x $HOME/bin/quietgcc

for i in lib9 libbio libmach_amd64 libregexp cmd pkg cmd/ebnflint cmd/gobuild cmd/godoc cmd/gofmt
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

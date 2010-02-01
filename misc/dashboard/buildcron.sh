#!/bin/sh
# Copyright 2010 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This script can be run to create a new builder and then
# to keep it running via cron.  First, run it by hand until it
# starts up without errors and can run the loop.  Then, once
# you're confident that it works, add this to your crontab:
#
#   */5 * * * *  cd $HOME; path/to/buildcron.sh darwin 386 >/dev/null 2>/dev/null

if [ $# != 2 ]; then
	echo 'usage: buildcron.sh goos goarch' 1>&2
	exit 2
fi

export GOOS=$1
export GOARCH=$2

# Check if we are already running.
# First command must not be pipeline, to avoid seeing extra processes in ps.
all=$(ps axwwu)
pid=$(echo "$all" | grep "buildcron.sh $1 $2" | grep -v "sh -c" | grep -v $$ | awk '{print $2}')
if [ "$pid" != "" ]; then
	#echo already running buildcron.sh $1 $2
	#echo "$all" | grep "buildcron.sh $1 $2" | grep -v "sh -c" | grep -v $$
	exit 0
fi

export BUILDHOST=godashboard.appspot.com
export BUILDER=${GOOS}-${GOARCH}
export GOROOT=$HOME/go-$BUILDER/go
export GOBIN=$HOME/go-$BUILDER/bin

if [ ! -f ~/.gobuildkey-$BUILDER ]; then
	echo "need gobuildkey for $BUILDER in ~/.gobuildkey-$BUILDER" 1>&2
	exit 2
fi

if [ ! -d $GOROOT ]; then
	mkdir -p $GOROOT
	hg clone https://go.googlecode.com/hg/ $GOROOT
fi
mkdir -p $GOROOT/bin

cd $GOROOT/..
cp go/misc/dashboard/builder.sh go/misc/dashboard/buildcontrol.py .
chmod a+x builder.sh buildcontrol.py
cd go
../buildcontrol.py next $BUILDER
cd ..
./builder.sh



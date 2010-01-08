#!/bin/sh

# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

fatal() {
    echo $1
    exit 1
}

if [ ! -d go ] ; then
    fatal "Please run in directory that contains a checked out repo in 'go'"
fi

if [ ! -f buildcontrol.py ] ; then
    fatal "Please include buildcontrol.py in this directory"
fi

if [ "x$BUILDER" == "x" ] ; then
    fatal "Please set \$BUILDER to the name of this builder"
fi

if [ "x$BUILDHOST" == "x" ] ; then
    fatal "Please set \$BUILDHOST to the hostname of the gobuild server"
fi

if [ "x$GOARCH" == "x" -o "x$GOOS" == "x" ] ; then
    fatal "Please set $GOARCH and $GOOS"
fi

export PATH=$PATH:`pwd`/candidate/bin
export GOBIN=`pwd`/candidate/bin

while true ; do
    cd go || fatal "Cannot cd into 'go'"
    hg pull -u || fatal "hg sync failed"
    rev=`python ../buildcontrol.py next $BUILDER`
    if [ $? -ne 0 ] ; then
        fatal "Cannot get next revision"
    fi
    cd .. || fatal "Cannot cd up"
    if [ "x$rev" == "x<none>" ] ; then
        sleep 10
        continue
    fi

    echo "Cloning for revision $rev"
    rm -Rf candidate
    hg clone -r $rev go candidate || fatal "hg clone failed"
    export GOROOT=`pwd`/candidate
    mkdir -p candidate/bin || fatal "Cannot create candidate/bin"
    cd candidate/src || fatal "Cannot cd into candidate/src"
    echo "Building revision $rev"
    ./all.bash > ../log 2>&1
    if [ $? -ne 0 ] ; then
        echo "Recording failure for $rev"
        python ../../buildcontrol.py record $BUILDER $rev ../log || fatal "Cannot record result"
    else
        echo "Recording success for $rev"
        python ../../buildcontrol.py record $BUILDER $rev ok || fatal "Cannot record result"
    fi
    cd ../.. || fatal "Cannot cd up"
done

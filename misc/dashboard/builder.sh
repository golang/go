#!/usr/bin/env bash

# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

fatal() {
    echo $0: $1 1>&2
    exit 1
}

if [ ! -d go ] ; then
    fatal "Please run in directory that contains a checked out repo in 'go'"
fi

if [ ! -f buildcontrol.py ] ; then
    fatal 'Please include buildcontrol.py in this directory'
fi

if [ "x$BUILDER" == "x" ] ; then
    fatal 'Please set $BUILDER to the name of this builder'
fi

if [ "x$BUILDHOST" == "x" ] ; then
    fatal 'Please set $BUILDHOST to the hostname of the gobuild server'
fi

if [ "x$GOARCH" == "x" -o "x$GOOS" == "x" ] ; then
    fatal 'Please set $GOARCH and $GOOS'
fi

export PATH=$PATH:`pwd`/candidate/bin
export GOBIN=`pwd`/candidate/bin
export GOROOT_FINAL=/usr/local/go

while true ; do (
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
    ALL=all.bash
    if [ -f all-$GOOS.bash ]; then
        ALL=all-$GOOS.bash
    elif [ -f all-$GOARCH.bash ]; then
        ALL=all-$GOARCH.bash
    fi
    ./$ALL > ../log 2>&1
    if [ $? -ne 0 ] ; then
        echo "Recording failure for $rev"
        python ../../buildcontrol.py record $BUILDER $rev ../log || fatal "Cannot record result"
    else
        echo "Recording success for $rev"
        python ../../buildcontrol.py record $BUILDER $rev ok || fatal "Cannot record result"
        if [ "$ALL" = "all.bash" ]; then
            echo "Running benchmarks"
            cd pkg || fatal "failed to cd to pkg"
            make bench > ../../benchmarks 2>&1
            python ../../../buildcontrol.py benchmarks $BUILDER $rev ../../benchmarks || fatal "Cannot record benchmarks"
            cd .. || fatal "failed to cd out of pkg"
        fi
        # check if we're at a release (via the hg summary)
        #  if so, package the tar.gz and upload to googlecode
        SUMMARY=$(hg log -l 1 | grep summary\: | awk '{print $2}')
        if [[ "x${SUMMARY:0:7}" == "xrelease" ]]; then
            echo "Uploading binary to googlecode"
            TARBALL="go.$SUMMARY.$BUILDER.tar.gz"
            ./clean.bash --nopkg
	    # move contents of candidate/ to candidate/go/ for archival
            cd ../..                     || fatal "Cannot cd up"
	    mv candidate go-candidate    || fatal "Cannot rename candidate"
	    mkdir candidate              || fatal "Cannot mkdir candidate"
	    mv go-candidate candidate/go || fatal "Cannot mv directory"
	    cd candidate                 || fatal "Cannot cd candidate"
	    # build tarball
            tar czf ../$TARBALL go       || fatal "Cannot create tarball"
            ../buildcontrol.py upload $BUILDER $SUMMARY ../$TARBALL || fatal "Cannot upload tarball"
        fi
    fi
    sleep 10
) done

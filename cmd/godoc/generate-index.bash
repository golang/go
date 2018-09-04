#!/usr/bin/env bash

# Copyright 2011 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This script creates a .zip file representing the $GOROOT file system
# and computes the corresponding search index files.
#
# These are used in production (see app.prod.yaml)

set -e -u -x

ZIPFILE=godoc.zip
INDEXFILE=godoc.index
SPLITFILES=index.split.

error() {
	echo "error: $1"
	exit 2
}

install() {
	go install
}

getArgs() {
	if [ ! -v GODOC_DOCSET ]; then
		GODOC_DOCSET="$(go env GOROOT)"
		echo "GODOC_DOCSET not set explicitly, using GOROOT instead"
	fi

	# safety checks
	if [ ! -d "$GODOC_DOCSET" ]; then
		error "$GODOC_DOCSET is not a directory"
	fi

	# reporting
	echo "GODOC_DOCSET = $GODOC_DOCSET"
}

makeZipfile() {
	echo "*** make $ZIPFILE"
	rm -f $ZIPFILE goroot
	ln -s "$GODOC_DOCSET" goroot
	zip -q -r $ZIPFILE goroot/* # glob to ignore dotfiles (like .git)
	rm goroot
}

makeIndexfile() {
	echo "*** make $INDEXFILE"
	godoc=$(go env GOPATH)/bin/godoc
	# NOTE: run godoc without GOPATH set. Otherwise third-party packages will end up in the index.
	GOPATH= $godoc -write_index -goroot goroot -index_files=$INDEXFILE -zip=$ZIPFILE
}

splitIndexfile() {
	echo "*** split $INDEXFILE"
	rm -f $SPLITFILES*
	split -b8m $INDEXFILE $SPLITFILES
}

cd $(dirname $0)

install
getArgs "$@"
makeZipfile
makeIndexfile
splitIndexfile
rm $INDEXFILE

echo "*** setup complete"

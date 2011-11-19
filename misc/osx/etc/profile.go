#!/bin/bash
# Copyright 2011 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# The output of this script will be eval'd by the user's shell on startup. This
# script decides what type of shell is being used in the same way as
# /usr/libexec/path_helper

if echo $SHELL | grep csh$ > /dev/null; then
	echo 'setenv GOROOT /usr/local/go'
else
	echo 'export GOROOT=/usr/local/go'
fi


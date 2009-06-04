#!/bin/bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

for i in cc 6l 6a 6c 8l 8a 8c 8g 5l 5a 5c 5g gc 6g ar db nm acid cov gobuild godefs prof gotest
do
	cd $i
	make clean
	cd ..
done

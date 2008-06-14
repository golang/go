#!/bin/bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

for i in 6l 6a 6c 6g gc cc db
do
	cd $i
	make clean
	cd ..
done

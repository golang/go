#!/bin/bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

for i in cmd runtime
do
	cd $i
	bash make.bash
	cd ..
done

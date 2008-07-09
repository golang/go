#!/bin/bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

bash clean.bash

for i in lib9 libbio libmach_amd64
do
	cd $i
	make install
	cd ..
done

for i in cmd runtime lib
do
	cd $i
	bash make.bash
	cd ..
done

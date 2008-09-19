# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

#!/bin/bash

rm -f $GOROOT/pkg/*

for i in os math net time
do
	cd $i
	make nuke
	cd ..
done


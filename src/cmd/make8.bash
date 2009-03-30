# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

#!/bin/bash

set -e

bash clean.bash

cd 8l
bash mkenam
make enam.o
cd ..

for i in cc 8l 8a 8c gc 8g ar db nm acid cov gobuild godefs prof gotest
do
	echo; echo; echo %%%% making $i %%%%; echo
	cd $i
	make install
	cd ..
done

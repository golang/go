# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

#!/bin/bash

set -e

for i in syscall os math
do
	echo; echo; echo %%%% making lib/$i %%%%; echo
	cd $i
	make install
	cd ..
done

# Don't sort the files in the for loop - some of the orderings matter.
rm -f *.6
for i in \
	fmt.go\
	flag.go\
	container/vector.go\
	rand.go\
	sort.go\
	io.go\
	bufio.go\
	strings.go\
	once.go\

do
	base=$(basename $i .go)
	echo 6g -o $GOROOT/pkg/$base.6 $i
	6g -o $GOROOT/pkg/$base.6 $i
done

for i in net time http
do
	echo; echo; echo %%%% making lib/$i %%%%; echo
	cd $i
	make install
	cd ..
done

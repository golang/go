# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

#!/bin/bash

echo; echo; echo %%%% making lib %%%%; echo

for i in os math
do
	echo; echo; echo %%%% making lib/$i %%%%; echo
	cd $i
	make install
	cd ..
done

rm -f *.6
for i in fmt.go flag.go container/vector.go rand.go sort.go strings.go
do
	base=$(basename $i .go)
	echo 6g -o $GOROOT/pkg/$base.6 $i
	6g -o $GOROOT/pkg/$base.6 $i
done


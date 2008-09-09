# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

#!/bin/bash

echo; echo; echo %%%% making lib %%%%; echo

rm -f *.6
for i in fmt.go flag.go container/vector.go sort.go
do
	base=$(basename $i .go)
	echo 6g -o $GOROOT/pkg/$base.6 $i
	6g -o $GOROOT/pkg/$base.6 $i
done

echo; echo; echo %%%% making lib/math %%%%; echo

cd math
bash make.bash
cd ..


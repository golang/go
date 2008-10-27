# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

#!/bin/bash

set -e

# clean
rm -f *.6 6.out test_integer

# integer package
6g integer.go
6g test_integer.go
6l -o test_integer integer.6 test_integer.6
./test_integer

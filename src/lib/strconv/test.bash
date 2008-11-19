#!/bin/bash
# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e
set -x

make clean
make
# make test
# ./test
# rm -f *.6 6.out test

#!/bin/bash
# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e
set -x

make clean
make
6g testatof.go
6l testatof.6
6.out
6g testftoa.go
6l testftoa.6
6.out
6g testfp.go
6l testfp.6
6.out
rm -f *.6 6.out

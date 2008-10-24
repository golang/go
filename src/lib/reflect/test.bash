#!/bin/bash
# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

make
6g test.go
6l test.6
6.out
rm -f *.6 6.out

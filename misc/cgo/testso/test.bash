#!/bin/sh
# Copyright 2011 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e
$(go env CC) $(go env GOGCCFLAGS) -shared -o libcgosotest.so cgoso_c.c
go build main.go
LD_LIBRARY_PATH=. ./main
rm -f libcgosotest.so main

#!/bin/sh
# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

if [ "$(uname)" = "FreeBSD" ]; then exit 0; fi

set -e
gomake hello fib chain
echo '*' hello >run.out
./hello >>run.out
echo '*' fib >>run.out
./fib >>run.out
echo '*' chain >>run.out
./chain >>run.out
diff run.out golden.out
gomake clean

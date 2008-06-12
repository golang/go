# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -ex

$HOME/bin/6a rt0_amd64_darwin.s
mv rt0_amd64_darwin.6 ../../lib/rt0_amd64_darwin.6

$HOME/bin/6a rt0_amd64_linux.s
mv rt0_amd64_linux.6 ../../lib/rt0_amd64_linux.6

$HOME/bin/6c runtime.c
mv runtime.6 ../../lib/rt_amd64.6

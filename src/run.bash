#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

eval $(go env)
export GOROOT   # the api test requires GOROOT to be set.

unset CDPATH	# in case user has it set
unset GOPATH    # we disallow local import for non-local packages, if $GOROOT happens
                # to be under $GOPATH, then some tests below will fail
unset GOBIN     # Issue 14340

export GOHOSTOS
export CC

# no core files, please
ulimit -c 0

# Raise soft limits to hard limits for NetBSD/OpenBSD.
# We need at least 256 files and ~300 MB of bss.
# On OS X ulimit -S -n rejects 'unlimited'.
#
# Note that ulimit -S -n may fail if ulimit -H -n is set higher than a
# non-root process is allowed to set the high limit.
# This is a system misconfiguration and should be fixed on the
# broken system, not "fixed" by ignoring the failure here.
# See longer discussion on golang.org/issue/7381. 
[ "$(ulimit -H -n)" == "unlimited" ] || ulimit -S -n $(ulimit -H -n)
[ "$(ulimit -H -d)" == "unlimited" ] || ulimit -S -d $(ulimit -H -d)

# Thread count limit on NetBSD 7.
if ulimit -T &> /dev/null; then
	[ "$(ulimit -H -T)" == "unlimited" ] || ulimit -S -T $(ulimit -H -T)
fi

exec go tool dist test -rebuild "$@"

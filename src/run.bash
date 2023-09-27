#!/usr/bin/env bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Environment variables that control run.bash:
#
# GO_TEST_SHARDS: number of "dist test" test shards that the
# $GOROOT/test directory will be sliced up into for parallel
# execution. Defaults to 1, unless GO_BUILDER_NAME is also specified,
# in which case it defaults to 10.
#
# GO_BUILDER_NAME: the name of the Go builder that's running the tests.
# Some tests are conditionally enabled or disabled based on the builder
# name or the builder name being non-empty.
#
# GO_TEST_SHORT: if set to a non-empty, false-ish string, run tests in "-short=false" mode.
# This environment variable is an internal implementation detail between the
# Go build system (x/build) and cmd/dist for the purpose of longtest builders,
# and will be removed if it stops being needed. See go.dev/issue/12508.
#
# GO_TEST_TIMEOUT_SCALE: a non-negative integer factor to scale test timeout by.
# Defaults to 1.

set -e

if [ ! -f ../bin/go ]; then
	echo 'run.bash must be run from $GOROOT/src after installing cmd/go' 1>&2
	exit 1
fi

export GOENV=off
eval $(../bin/go tool dist env)

unset CDPATH	# in case user has it set

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
[ "$(ulimit -H -n)" = "unlimited" ] || ulimit -S -n $(ulimit -H -n)
[ "$(ulimit -H -d)" = "unlimited" ] || ulimit -S -d $(ulimit -H -d)

# Thread count limit on NetBSD 7.
if ulimit -T &> /dev/null; then
	[ "$(ulimit -H -T)" = "unlimited" ] || ulimit -S -T $(ulimit -H -T)
fi

export GOPATH=/nonexist-gopath
exec ../bin/go tool dist test -rebuild "$@"

#!/usr/bin/env bash
# Copyright 2016 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# mkcanned.bash creates canned traces for the trace test suite using
# the current Go version.

set -e

if [ $# != 1 ]; then
    echo "usage: $0 <label>" >&2
    exit 1
fi

go test -run ClientServerParallel4 -trace "testdata/http_$1_good" net/http
go test -run 'TraceStress$|TraceStressStartStop$|TestUserTaskSpan$' runtime/trace -savetraces
mv ../../runtime/trace/TestTraceStress.trace "testdata/stress_$1_good"
mv ../../runtime/trace/TestTraceStressStartStop.trace "testdata/stress_start_stop_$1_good"
mv ../../runtime/trace/TestUserTaskSpan.trace "testdata/user_task_span_$1_good"

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace_test

import (
	"bytes"
	"flag"
	. "runtime/trace"
	"testing"
	"time"
)

var dumpTraces = flag.Bool("dump-traces", false, "dump traces to a file, even on success")

// This file just contains smoke tests and tests of runtime/trace logic only.
// It doesn't validate the resulting traces. See the internal/trace package for
// more comprehensive end-to-end tests.

func TestTraceStartStop(t *testing.T) {
	if IsEnabled() {
		t.Skip("skipping because -test.trace is set")
	}
	buf := new(bytes.Buffer)
	if err := Start(buf); err != nil {
		t.Fatalf("failed to start tracing: %v", err)
	}
	Stop()
	size := buf.Len()
	if size == 0 {
		t.Fatalf("trace is empty")
	}
	time.Sleep(100 * time.Millisecond)
	if size != buf.Len() {
		t.Fatalf("trace writes after stop: %v -> %v", size, buf.Len())
	}
}

func TestTraceDoubleStart(t *testing.T) {
	if IsEnabled() {
		t.Skip("skipping because -test.trace is set")
	}
	Stop()
	buf := new(bytes.Buffer)
	if err := Start(buf); err != nil {
		t.Fatalf("failed to start tracing: %v", err)
	}
	if err := Start(buf); err == nil {
		t.Fatalf("succeed to start tracing second time")
	}
	Stop()
	Stop()
}

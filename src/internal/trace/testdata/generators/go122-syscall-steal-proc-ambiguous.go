// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests syscall P stealing.
//
// Specifically, it tests a scenario wherein, without a
// P sequence number of GoSyscallBegin, the syscall that
// a ProcSteal applies to is ambiguous. This only happens in
// practice when the events aren't already properly ordered
// by timestamp, since the ProcSteal won't be seen until after
// the correct GoSyscallBegin appears on the frontier.

package main

import (
	"internal/trace"
	"internal/trace/internal/testgen"
	"internal/trace/tracev2"
	"internal/trace/version"
)

func main() {
	testgen.Main(version.Go122, gen)
}

func gen(t *testgen.Trace) {
	t.DisableTimestamps()

	g := t.Generation(1)

	// One goroutine does a syscall without blocking, then another one where
	// it's P gets stolen.
	b0 := g.Batch(trace.ThreadID(0), 0)
	b0.Event("ProcStatus", trace.ProcID(0), tracev2.ProcRunning)
	b0.Event("GoStatus", trace.GoID(1), trace.ThreadID(0), tracev2.GoRunning)
	b0.Event("GoSyscallBegin", testgen.Seq(1), testgen.NoStack)
	b0.Event("GoSyscallEnd")
	b0.Event("GoSyscallBegin", testgen.Seq(2), testgen.NoStack)
	b0.Event("GoSyscallEndBlocked")

	// A running goroutine steals proc 0.
	b1 := g.Batch(trace.ThreadID(1), 0)
	b1.Event("ProcStatus", trace.ProcID(2), tracev2.ProcRunning)
	b1.Event("GoStatus", trace.GoID(2), trace.ThreadID(1), tracev2.GoRunning)
	b1.Event("ProcSteal", trace.ProcID(0), testgen.Seq(3), trace.ThreadID(0))
}

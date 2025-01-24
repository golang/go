// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests syscall P stealing.
//
// Specifically, it tests a scenario where a thread 'steals'
// a P from itself. It's just a ProcStop with extra steps when
// it happens on the same P.

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

	// A goroutine execute a syscall and steals its own P, then starts running
	// on that P.
	b0 := g.Batch(trace.ThreadID(0), 0)
	b0.Event("ProcStatus", trace.ProcID(0), tracev2.ProcRunning)
	b0.Event("GoStatus", trace.GoID(1), trace.ThreadID(0), tracev2.GoRunning)
	b0.Event("GoSyscallBegin", testgen.Seq(1), testgen.NoStack)
	b0.Event("ProcSteal", trace.ProcID(0), testgen.Seq(2), trace.ThreadID(0))
	b0.Event("ProcStart", trace.ProcID(0), testgen.Seq(3))
	b0.Event("GoSyscallEndBlocked")
}

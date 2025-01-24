// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests syscall P stealing.

package main

import (
	"internal/trace"
	"internal/trace/internal/testgen"
	"internal/trace/tracev2"
)

func main() {
	testgen.Main(gen)
}

func gen(t *testgen.Trace) {
	g := t.Generation(1)

	// One goroutine enters a syscall, grabs a P, and starts running.
	b0 := g.Batch(trace.ThreadID(0), 0)
	b0.Event("ProcStatus", trace.ProcID(1), tracev2.ProcIdle)
	b0.Event("ProcStatus", trace.ProcID(0), tracev2.ProcRunning)
	b0.Event("GoStatus", trace.GoID(1), trace.ThreadID(0), tracev2.GoRunning)
	b0.Event("GoSyscallBegin", testgen.Seq(1), testgen.NoStack)
	b0.Event("ProcStart", trace.ProcID(1), testgen.Seq(1))
	b0.Event("GoSyscallEndBlocked")

	// A bare M steals the goroutine's P.
	b1 := g.Batch(trace.ThreadID(1), 0)
	b1.Event("ProcSteal", trace.ProcID(0), testgen.Seq(2), trace.ThreadID(0))
}

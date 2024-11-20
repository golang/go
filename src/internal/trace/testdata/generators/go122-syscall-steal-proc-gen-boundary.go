// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests syscall P stealing at a generation boundary.

package main

import (
	"internal/trace"
	"internal/trace/event/go122"
	testgen "internal/trace/internal/testgen/go122"
)

func main() {
	testgen.Main(gen)
}

func gen(t *testgen.Trace) {
	g := t.Generation(1)

	// One goroutine is exiting with a syscall. It already
	// acquired a new P.
	b0 := g.Batch(trace.ThreadID(0), 0)
	b0.Event("ProcStatus", trace.ProcID(1), go122.ProcRunning)
	b0.Event("GoStatus", trace.GoID(1), trace.ThreadID(0), go122.GoSyscall)
	b0.Event("GoSyscallEndBlocked")

	// A running goroutine stole P0 at the generation boundary.
	b1 := g.Batch(trace.ThreadID(1), 0)
	b1.Event("ProcStatus", trace.ProcID(2), go122.ProcRunning)
	b1.Event("GoStatus", trace.GoID(2), trace.ThreadID(1), go122.GoRunning)
	b1.Event("ProcStatus", trace.ProcID(0), go122.ProcSyscallAbandoned)
	b1.Event("ProcSteal", trace.ProcID(0), testgen.Seq(1), trace.ThreadID(0))
}

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests syscall P stealing from a goroutine and thread
// that have been in a syscall the entire generation.

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

	// Steal proc from a goroutine that's been blocked
	// in a syscall the entire generation.
	b0 := g.Batch(trace.ThreadID(0), 0)
	b0.Event("ProcStatus", trace.ProcID(0), go122.ProcSyscallAbandoned)
	b0.Event("ProcSteal", trace.ProcID(0), testgen.Seq(1), trace.ThreadID(1))

	// Status event for a goroutine blocked in a syscall for the entire generation.
	bz := g.Batch(trace.NoThread, 0)
	bz.Event("GoStatus", trace.GoID(1), trace.ThreadID(1), go122.GoSyscall)
}

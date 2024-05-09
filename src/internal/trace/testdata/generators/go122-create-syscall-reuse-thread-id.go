// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests a G being created from within a syscall.
//
// Specifically, it tests a scenerio wherein a C
// thread is calling into Go, creating a goroutine in
// a syscall (in the tracer's model). The system is free
// to reuse thread IDs, so first a thread ID is used to
// call into Go, and then is used for a Go-created thread.
//
// This is a regression test. The trace parser didn't correctly
// model GoDestroySyscall as dropping its P (even if the runtime
// did). It turns out this is actually fine if all the threads
// in the trace have unique IDs, since the P just stays associated
// with an eternally dead thread, and it's stolen by some other
// thread later. But if thread IDs are reused, then the tracer
// gets confused when trying to advance events on the new thread.
// The now-dead thread which exited on a GoDestroySyscall still has
// its P associated and this transfers to the newly-live thread
// in the parser's state because they share a thread ID.

package main

import (
	"internal/trace/v2"
	"internal/trace/v2/event/go122"
	testgen "internal/trace/v2/internal/testgen/go122"
)

func main() {
	testgen.Main(gen)
}

func gen(t *testgen.Trace) {
	g := t.Generation(1)

	// A C thread calls into Go and acquires a P. It returns
	// back to C, destroying the G.
	b0 := g.Batch(trace.ThreadID(0), 0)
	b0.Event("GoCreateSyscall", trace.GoID(4))
	b0.Event("GoSyscallEndBlocked")
	b0.Event("ProcStatus", trace.ProcID(0), go122.ProcIdle)
	b0.Event("ProcStart", trace.ProcID(0), testgen.Seq(1))
	b0.Event("GoStatus", trace.GoID(4), trace.NoThread, go122.GoRunnable)
	b0.Event("GoStart", trace.GoID(4), testgen.Seq(1))
	b0.Event("GoSyscallBegin", testgen.Seq(2), testgen.NoStack)
	b0.Event("GoDestroySyscall")

	// A new Go-created thread with the same ID appears and
	// starts running, then tries to steal the P from the
	// first thread. The stealing is interesting because if
	// the parser handles GoDestroySyscall wrong, then we
	// have a self-steal here potentially that doesn't make
	// sense.
	b1 := g.Batch(trace.ThreadID(0), 0)
	b1.Event("ProcStatus", trace.ProcID(1), go122.ProcIdle)
	b1.Event("ProcStart", trace.ProcID(1), testgen.Seq(1))
	b1.Event("ProcSteal", trace.ProcID(0), testgen.Seq(3), trace.ThreadID(0))
}

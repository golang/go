// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests overlapping syscalls that release the same P.

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
	g := t.Generation(1)
	b0 := g.Batch(trace.ThreadID(0), 1)
	b0.Event("ProcStatus", trace.ProcID(0), tracev2.ProcRunning)
	b0.Event("GoStatus", trace.GoID(1), trace.ThreadID(0), tracev2.GoRunning)
	b0.Event("GoCreate", trace.GoID(2), testgen.NoStack, testgen.NoStack)
	b0.Event("GoSyscallBegin", testgen.Seq(1), testgen.NoStack)

	// G2 acquires G1's P and enters a syscall before G1 returns.
	b1 := g.Batch(trace.ThreadID(1), 6)
	b1.Event("ProcSteal", trace.ProcID(0), testgen.Seq(2), trace.ThreadID(0))
	b1.Event("ProcStart", trace.ProcID(0), testgen.Seq(3))
	b1.Event("GoStart", trace.GoID(2), testgen.Seq(1))
	b1.Event("GoSyscallBegin", testgen.Seq(4), testgen.NoStack)

	b0 = g.Batch(trace.ThreadID(0), 11)
	b0.Event("GoSyscallEndBlocked")

	// G2's P is stolen after G1 has returned from its syscall.
	b2 := g.Batch(trace.ThreadID(2), 13)
	b2.Event("ProcSteal", trace.ProcID(0), testgen.Seq(5), trace.ThreadID(1))
	b2.Event("ProcStart", trace.ProcID(0), testgen.Seq(6))

	b1 = g.Batch(trace.ThreadID(1), 16)
	b1.Event("GoSyscallEndBlocked")
}

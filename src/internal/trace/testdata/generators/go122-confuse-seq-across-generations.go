// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Regression test for an issue found in development.
//
// The core of the issue is that if generation counters
// aren't considered as part of sequence numbers, then
// it's possible to accidentally advance without a
// GoStatus event.
//
// The situation is one in which it just so happens that
// an event on the frontier for a following generation
// has a sequence number exactly one higher than the last
// sequence number for e.g. a goroutine in the previous
// generation. The parser should wait to find a GoStatus
// event before advancing into the next generation at all.
// It turns out this situation is pretty rare; the GoStatus
// event almost always shows up first in practice. But it
// can and did happen.

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
	g1 := t.Generation(1)

	// A running goroutine blocks.
	b10 := g1.Batch(trace.ThreadID(0), 0)
	b10.Event("ProcStatus", trace.ProcID(0), tracev2.ProcRunning)
	b10.Event("GoStatus", trace.GoID(1), trace.ThreadID(0), tracev2.GoRunning)
	b10.Event("GoStop", "whatever", testgen.NoStack)

	// The running goroutine gets unblocked.
	b11 := g1.Batch(trace.ThreadID(1), 0)
	b11.Event("ProcStatus", trace.ProcID(1), tracev2.ProcRunning)
	b11.Event("GoStart", trace.GoID(1), testgen.Seq(1))
	b11.Event("GoStop", "whatever", testgen.NoStack)

	g2 := t.Generation(2)

	// Start running the goroutine, but later.
	b21 := g2.Batch(trace.ThreadID(1), 3)
	b21.Event("ProcStatus", trace.ProcID(1), tracev2.ProcRunning)
	b21.Event("GoStart", trace.GoID(1), testgen.Seq(2))

	// The goroutine starts running, then stops, then starts again.
	b20 := g2.Batch(trace.ThreadID(0), 5)
	b20.Event("ProcStatus", trace.ProcID(0), tracev2.ProcRunning)
	b20.Event("GoStatus", trace.GoID(1), trace.ThreadID(0), tracev2.GoRunnable)
	b20.Event("GoStart", trace.GoID(1), testgen.Seq(1))
	b20.Event("GoStop", "whatever", testgen.NoStack)
}

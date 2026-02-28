// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Regression test for an issue found in development.
//
// GoCreate events can happen on bare Ps in a variety of situations and
// and earlier version of the parser assumed this wasn't possible. At
// the time of writing, one such example is goroutines created by expiring
// timers.

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
	g1 := t.Generation(1)

	// A goroutine gets created on a running P, then starts running.
	b0 := g1.Batch(trace.ThreadID(0), 0)
	b0.Event("ProcStatus", trace.ProcID(0), tracev2.ProcRunning)
	b0.Event("GoCreate", trace.GoID(5), testgen.NoStack, testgen.NoStack)
	b0.Event("GoStart", trace.GoID(5), testgen.Seq(1))
	b0.Event("GoStop", "whatever", testgen.NoStack)
}

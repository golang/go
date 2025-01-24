// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Regression test for an issue found in development.
//
// The issue is that EvUserTaskEnd events don't carry the
// task type with them, so the parser needs to track that
// information. But if the parser just tracks the string ID
// and not the string itself, that string ID may not be valid
// for use in future generations.

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

	// A running goroutine emits a task begin.
	b1 := g1.Batch(trace.ThreadID(0), 0)
	b1.Event("ProcStatus", trace.ProcID(0), tracev2.ProcRunning)
	b1.Event("GoStatus", trace.GoID(1), trace.ThreadID(0), tracev2.GoRunning)
	b1.Event("UserTaskBegin", trace.TaskID(2), trace.TaskID(0) /* 0 means no parent, not background */, "my task", testgen.NoStack)

	g2 := t.Generation(2)

	// That same goroutine emits a task end in the following generation.
	b2 := g2.Batch(trace.ThreadID(0), 5)
	b2.Event("ProcStatus", trace.ProcID(0), tracev2.ProcRunning)
	b2.Event("GoStatus", trace.GoID(1), trace.ThreadID(0), tracev2.GoRunning)
	b2.Event("UserTaskEnd", trace.TaskID(2), testgen.NoStack)
}

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"internal/trace"
	"internal/trace/internal/testgen"
	"internal/trace/tracev2"
	"internal/trace/version"
	"time"
)

func main() {
	testgen.Main(version.Go123, gen)
}

func gen(t *testgen.Trace) {
	g1 := t.Generation(1)
	g1.Sync(1000000000, 10, 0, time.Time{})
	b10 := g1.Batch(1, 15)
	b10.Event("ProcStatus", trace.ProcID(0), tracev2.ProcRunning)
	g2 := t.Generation(2)
	g2.Sync(500000000, 20, 0, time.Time{})
	g3 := t.Generation(3)
	b30 := g3.Batch(1, 30)
	b30.Event("ProcStatus", trace.ProcID(0), tracev2.ProcRunning)
	g3.Sync(500000000, 40, 0, time.Time{})
}

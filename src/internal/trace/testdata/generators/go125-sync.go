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
	testgen.Main(version.Go125, gen)
}

func gen(t *testgen.Trace) {
	start := time.Date(2025, 2, 28, 15, 4, 9, 123, time.UTC)
	g1 := t.Generation(1)
	g1.Sync(1000000000, 10, 99, start)
	b10 := g1.Batch(1, 15)
	b10.Event("ProcStatus", trace.ProcID(0), tracev2.ProcRunning)
	g2 := t.Generation(2)
	g2.Sync(500000000, 20, 199, start.Add(1*time.Second))
	g3 := t.Generation(3)
	g3.Sync(500000000, 30, 299, start.Add(2*time.Second))
	b30 := g3.Batch(1, 40)
	b30.Event("ProcStatus", trace.ProcID(0), tracev2.ProcRunning)
}

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace_test

import (
	"context"
	"io"
	. "runtime/trace"
	"strings"
	"testing"
)

func TestStartRegionLongString(t *testing.T) {
	// Regression test: a region name longer than the trace region
	// allocator's block size (~64KB) used to crash with
	// "traceRegion: alloc too large" because traceStringTable.put
	// inserted the full string into the trace map before truncation.
	Start(io.Discard)
	defer Stop()

	big := strings.Repeat("x", 70_000)
	StartRegion(context.Background(), big).End()
}

func BenchmarkStartRegion(b *testing.B) {
	b.ReportAllocs()
	ctx, task := NewTask(context.Background(), "benchmark")
	defer task.End()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			StartRegion(ctx, "region").End()
		}
	})
}

func BenchmarkNewTask(b *testing.B) {
	b.ReportAllocs()
	pctx, task := NewTask(context.Background(), "benchmark")
	defer task.End()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, task := NewTask(pctx, "task")
			task.End()
		}
	})
}

func BenchmarkLog(b *testing.B) {
	b.ReportAllocs()

	Start(io.Discard)
	defer Stop()

	ctx := context.Background()
	for b.Loop() {
		Log(ctx, "", "")
	}
}

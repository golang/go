// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace_test

import (
	"context"
	. "runtime/trace"
	"testing"
)

func BenchmarkStartRegion(b *testing.B) {
	b.ReportAllocs()
	ctx, task := NewTask(context.Background(), "benchmark")
	defer task.End()

	b.RunParallel(func { pb -> for pb.Next() {
		StartRegion(ctx, "region").End()
	} })
}

func BenchmarkNewTask(b *testing.B) {
	b.ReportAllocs()
	pctx, task := NewTask(context.Background(), "benchmark")
	defer task.End()

	b.RunParallel(func { pb -> for pb.Next() {
		_, task := NewTask(pctx, "task")
		task.End()
	} })
}

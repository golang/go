// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync_test

// BenchmarkHotLoad (M22-S4, SPEC T-206, Table T-PERF "Hot[T] 0ns RCU read")
// is report-only. A 0 ns read is not measurable; the benchmark reports what
// one Hot.Load costs next to a plain atomic.Pointer load of the same value.
// Hot[T] stores through atomic.Pointer[T] and Load is one atomic load, so
// the two columns are the same load with and without the wrapper. The serial
// pair measures one read; the parallel pair has every processor read the one
// value, which shows whether readers contend: a load takes no lock and
// writes nothing. Each load is asserted allocation-free first, because a
// read that allocates is not an atomic load. Every loop calls the method
// directly, so the compiler inlines Hot.Load as it does at a call site.

import (
	"sync/atomic"
	"testing"

	logosync "logos/sync"
)

var hotSink *hotPoint

func BenchmarkHotLoad(b *testing.B) {
	h := new(logosync.Hot[hotPoint])
	h.Store(&hotPoint{x: 1, y: 1})
	var plain atomic.Pointer[hotPoint]
	plain.Store(&hotPoint{x: 1, y: 1})

	if n := testing.AllocsPerRun(100, func() { hotSink = h.Load() }); n != 0 {
		b.Fatalf("Hot.Load allocates %v times per call, want 0", n)
	}
	if n := testing.AllocsPerRun(100, func() { hotSink = plain.Load() }); n != 0 {
		b.Fatalf("atomic.Pointer.Load allocates %v times per call, want 0", n)
	}

	b.Run("hot", func(b *testing.B) {
		b.ReportAllocs()
		var sink *hotPoint
		for b.Loop() {
			sink = h.Load()
		}
		hotSink = sink
	})
	b.Run("atomic-pointer", func(b *testing.B) {
		b.ReportAllocs()
		var sink *hotPoint
		for b.Loop() {
			sink = plain.Load()
		}
		hotSink = sink
	})
	b.Run("hot-parallel", func(b *testing.B) {
		b.ReportAllocs()
		b.RunParallel(func(pb *testing.PB) {
			var sink *hotPoint
			for pb.Next() {
				sink = h.Load()
			}
			_ = sink
		})
	})
	b.Run("atomic-pointer-parallel", func(b *testing.B) {
		b.ReportAllocs()
		b.RunParallel(func(pb *testing.PB) {
			var sink *hotPoint
			for pb.Next() {
				sink = plain.Load()
			}
			_ = sink
		})
	})
}

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.greenteagc && (!js || !wasm)

package runtime_test

import (
	"math/rand/v2"
	"runtime"
	"runtime/debug"
	"slices"
	"testing"
	"unsafe"
)

type spanOrderRec struct {
	near, far *spanOrderRec
	pad       [6]uintptr
}

// spanOrderHeap allocates n records in allocation order. Each points to a
// neighbour within 32 records (the locality of a graph built in traversal
// order) and to one record chosen at random (the long edge).
func spanOrderHeap(n int) []*spanOrderRec {
	rng := rand.New(rand.NewPCG(19, 6))
	recs := make([]*spanOrderRec, n)
	for i := range recs {
		recs[i] = new(spanOrderRec)
	}
	for i, r := range recs {
		r.near = recs[(i+rng.IntN(65)-32+n)%n]
		r.far = recs[rng.IntN(n)]
	}
	return recs
}

// spanOrderSnapshot records the two edges of every record.
func spanOrderSnapshot(recs []*spanOrderRec) (near, far []*spanOrderRec) {
	near, far = make([]*spanOrderRec, len(recs)), make([]*spanOrderRec, len(recs))
	for i, r := range recs {
		near[i], far[i] = r.near, r.far
	}
	return near, far
}

func checkSpanOrderHeap(tb testing.TB, recs []*spanOrderRec, near, far []*spanOrderRec) {
	tb.Helper()
	for i, r := range recs {
		if r.near != near[i] || r.far != far[i] {
			tb.Fatalf("record %d changed", i)
		}
	}
}

// BenchmarkLogosSpanOrder scans the same dirty spans with the collector's own
// span scan, once in ascending address order (the span deck) and once in LIFO
// order of a random discovery sequence (a workbuf stack reaching spans in
// graph order), and reports the user-mode L1 data cache read misses per span
// next to the time per span. The counts are recorded, not a pass condition.
//
// The rig sets and clears mark bits itself, so the benchmark assumes no
// collection runs while they are set: it disables the collector with
// SetGCPercent(-1) around every timed loop and restores it before it checks
// that every record survived.
func BenchmarkLogosSpanOrder(b *testing.B) {
	const records = 1 << 19
	recs := spanOrderHeap(records)
	near, far := spanOrderSnapshot(recs)
	objs := make([]unsafe.Pointer, records)
	for i, r := range recs {
		objs[i] = unsafe.Pointer(r)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	runtime.GC()
	oldPercent := debug.SetGCPercent(-1)
	defer debug.SetGCPercent(oldPercent)
	rig := runtime.NewLogosSpanOrderRig(objs)
	if rig == nil {
		b.Skip("the records did not land in clean spans with inline mark bits")
	}

	address := slices.Clone(rig.Bases())
	slices.Sort(address)
	lifo := slices.Clone(address)
	rand.New(rand.NewPCG(19, 7)).Shuffle(len(lifo), func(i, j int) { lifo[i], lifo[j] = lifo[j], lifo[i] })
	slices.Reverse(lifo)

	for _, order := range []struct {
		name  string
		spans []uintptr
	}{{"address", address}, {"lifo", lifo}} {
		b.Run(order.name, func(b *testing.B) {
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()
			counter, err := openL1Misses()
			if err != nil {
				b.Logf("L1 misses not reported: %v", err)
			} else {
				defer counter.close()
			}
			var misses uint64
			counted := counter != nil
			for range b.N {
				b.StopTimer()
				rig.Arm()
				b.StartTimer()
				if counted {
					counter.start()
				}
				rig.Scan(order.spans)
				if counted {
					n, ok := counter.stop()
					misses += n
					counted = ok
				}
				b.StopTimer()
				rig.Disarm()
				b.StartTimer()
			}
			perSpan := float64(b.N) * float64(len(order.spans))
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/perSpan, "ns/span")
			if counted {
				b.ReportMetric(float64(misses)/perSpan, "L1miss/span")
			}
		})
	}
	checkSpanOrderHeap(b, recs, near, far)
	debug.SetGCPercent(oldPercent)
	runtime.GC()
	checkSpanOrderHeap(b, recs, near, far)
	runtime.KeepAlive(objs)
}

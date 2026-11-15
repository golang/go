// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlogos

import "testing"

func BenchmarkFastcallVoid(b *testing.B) {
	for b.Loop() {
		callBenchVoidFast()
	}
}

func BenchmarkSlowcallVoid(b *testing.B) {
	for b.Loop() {
		callBenchVoidSlow()
	}
}

func BenchmarkFastcallOne(b *testing.B) {
	var sink int32
	for b.Loop() {
		sink = callBenchOneFast(sink & 1023)
	}
	_ = sink
}

func BenchmarkSlowcallOne(b *testing.B) {
	var sink int32
	for b.Loop() {
		sink = callBenchOneSlow(sink & 1023)
	}
	_ = sink
}

func BenchmarkFastcallThree(b *testing.B) {
	var sink int32
	for b.Loop() {
		sink = callBenchThreeFast(sink&1023, 1, 2)
	}
	_ = sink
}

func BenchmarkSlowcallThree(b *testing.B) {
	var sink int32
	for b.Loop() {
		sink = callBenchThreeSlow(sink&1023, 1, 2)
	}
	_ = sink
}

func BenchmarkFastcallPtr2(b *testing.B) {
	a := int32(3)
	c := int32(4)
	var sink int32
	for b.Loop() {
		sink = callBenchRead2Fast(&a, &c)
	}
	_ = sink
}

func BenchmarkSlowcallPtr2(b *testing.B) {
	a := int32(3)
	c := int32(4)
	var sink int32
	for b.Loop() {
		sink = callBenchRead2Slow(&a, &c)
	}
	_ = sink
}

func BenchmarkFastcallSlice2(b *testing.B) {
	p := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	q := []byte{9, 10, 11, 12, 13, 14, 15, 16}
	var sink int32
	for b.Loop() {
		sink = callBenchSum2Fast(p, q)
	}
	_ = sink
}

func BenchmarkSlowcallSlice2(b *testing.B) {
	p := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	q := []byte{9, 10, 11, 12, 13, 14, 15, 16}
	var sink int32
	for b.Loop() {
		sink = callBenchSum2Slow(p, q)
	}
	_ = sink
}

func BenchmarkFastcallErrno(b *testing.B) {
	var r int32
	var err error
	for b.Loop() {
		r, err = callBenchMayfailFast(0)
	}
	_, _ = r, err
}

func BenchmarkSlowcallErrno(b *testing.B) {
	var r int32
	var err error
	for b.Loop() {
		r, err = callBenchMayfailSlow(0)
	}
	_, _ = r, err
}

func BenchmarkFastcallStruct(b *testing.B) {
	p, _ := benchPointVals()
	var sink int32
	for b.Loop() {
		sink = callBenchPointFast(p)
	}
	_ = sink
}

func BenchmarkSlowcallStruct(b *testing.B) {
	_, q := benchPointVals()
	var sink int32
	for b.Loop() {
		sink = callBenchPointSlow(q)
	}
	_ = sink
}

const benchBurnN = 512

func BenchmarkFastcallBurn(b *testing.B) {
	var sink int32
	for b.Loop() {
		sink = callBenchBurnFast(benchBurnN)
	}
	_ = sink
}

func BenchmarkSlowcallBurn(b *testing.B) {
	var sink int32
	for b.Loop() {
		sink = callBenchBurnSlow(benchBurnN)
	}
	_ = sink
}

const benchFloorInner = 2000

func BenchmarkFloorEmpty(b *testing.B) {
	var sink int64
	for b.Loop() {
		sink = callBenchFloorEmpty(benchFloorInner)
	}
	_ = sink
}

func BenchmarkFloorCall(b *testing.B) {
	var sink int64
	for b.Loop() {
		sink = callBenchFloorCall(benchFloorInner)
	}
	_ = sink
}

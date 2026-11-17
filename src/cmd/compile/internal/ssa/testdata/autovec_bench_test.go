// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package autovecbench

import "testing"

const autovecBenchSize = 8192

var (
	autovecBenchInput16  [autovecBenchSize]int16
	autovecBenchOutput16 [autovecBenchSize]int16
	autovecBenchInput32  [autovecBenchSize]uint32
	autovecBenchOutput32 [autovecBenchSize]uint32
	autovecBenchSink     uint64
)

func init() {
	for i := range autovecBenchInput16 {
		autovecBenchInput16[i] = int16(uint16(uint32(i)*40503 + 12345))
		autovecBenchInput32[i] = uint32(i)*0x9e3779b9 + 0x80000001
	}
}

//go:noinline
func benchFixedPoint(dst, src []int16) {
	if len(dst) > len(src) {
		return
	}
	for i := range dst {
		v := src[i]
		dst[i] = ((v*3 + 7) >> 4) + ^v - v
	}
}

//go:noinline
func benchStartFixed(dst, src []uint32) {
	if len(dst) > len(src) {
		return
	}
	for i := 14; i < len(dst); i++ {
		dst[i] = src[i] + 37
	}
}

//go:noinline
func benchStartVariable(dst, src []uint32, start int) {
	if start < 0 || start > len(dst) || len(dst) > len(src) {
		return
	}
	for i := start; i < len(dst); i++ {
		dst[i] = src[i] + 37
	}
}

//go:noinline
func benchScalarFixedPoint(v int16) int16 { return ((v*3 + 7) >> 4) + ^v - v }

//go:noinline
func benchScalarStart(v uint32) uint32 { return v + 37 }

func TestAutovecBenchmarkParity(t *testing.T) {
	src16, dst16 := autovecBenchInput16[:], autovecBenchOutput16[:]
	benchFixedPoint(dst16, src16)
	for i, got := range dst16 {
		if want := benchScalarFixedPoint(src16[i]); got != want {
			t.Fatalf("fixed-point lane %d: got %d, want %d", i, got, want)
		}
	}
	src32, dst32 := autovecBenchInput32[:], autovecBenchOutput32[:]
	for i := range dst32 {
		dst32[i] = ^uint32(0)
	}
	benchStartFixed(dst32, src32)
	for i, got := range dst32 {
		want := ^uint32(0)
		if i >= 14 {
			want = benchScalarStart(src32[i])
		}
		if got != want {
			t.Fatalf("fixed start lane %d: got %d, want %d", i, got, want)
		}
	}
	for i := range dst32 {
		dst32[i] = ^uint32(0)
	}
	benchStartVariable(dst32, src32, 14)
	for i, got := range dst32 {
		want := ^uint32(0)
		if i >= 14 {
			want = benchScalarStart(src32[i])
		}
		if got != want {
			t.Fatalf("variable start lane %d: got %d, want %d", i, got, want)
		}
	}
}

func BenchmarkAutovecFixedPoint(b *testing.B) {
	dst, src := autovecBenchOutput16[:], autovecBenchInput16[:]
	b.SetBytes(4 * int64(len(dst)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchFixedPoint(dst, src)
	}
	b.StopTimer()
	autovecBenchSink = uint64(uint16(dst[len(dst)-1]))
}

func BenchmarkAutovecStartFixed(b *testing.B) {
	dst, src := autovecBenchOutput32[:], autovecBenchInput32[:]
	b.SetBytes(8 * int64(len(dst)-14))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStartFixed(dst, src)
	}
	b.StopTimer()
	autovecBenchSink = uint64(dst[len(dst)-1])
}

func BenchmarkAutovecStartVariable(b *testing.B) {
	dst, src := autovecBenchOutput32[:], autovecBenchInput32[:]
	start := 14
	b.SetBytes(8 * int64(len(dst)-start))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStartVariable(dst, src, start)
	}
	b.StopTimer()
	autovecBenchSink = uint64(dst[len(dst)-1])
}

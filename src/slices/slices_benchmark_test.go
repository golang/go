// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slices

import (
	"testing"
)

var sink any = nil

func BenchmarkIntClone_10B(b *testing.B) {
	benchmarkClone[int](b, 10)
}

func BenchmarkIntClone_1Kb(b *testing.B) {
	benchmarkClone[int](b, 1<<10)
}

func BenchmarkIntClone_10Kb(b *testing.B) {
	benchmarkClone[int](b, 10<<10)
}

func BenchmarkIntClone_1Mb(b *testing.B) {
	benchmarkClone[int](b, 1<<20)
}

func BenchmarkIntClone_10Mb(b *testing.B) {
	benchmarkClone[int](b, 10<<20)
}

func BenchmarkByteClone_10B(b *testing.B) {
	benchmarkClone[byte](b, 10)
}

func BenchmarkByteClone_10Kb(b *testing.B) {
	benchmarkClone[byte](b, 10<<10)
}

func benchmarkClone[T int | byte](b *testing.B, n int) {
	s1 := make([]T, n)
	for i := 0; i < n/2; i++ {
		s1[i] = T(i)
		s1[n-i-1] = T(i * 2)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sink = Clone(s1)
	}
	if sink == nil {
		b.Fatal("Benchmark did not run!")
	}
	sink = nil
}
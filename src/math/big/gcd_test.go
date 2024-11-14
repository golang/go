// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements a GCD benchmark.
// Usage: go test math/big -test.bench GCD

package big

import (
	"math/rand"
	"testing"
)

// randInt returns a pseudo-random Int in the range [1<<(size-1), (1<<size) - 1]
func randInt(r *rand.Rand, size uint) *Int {
	n := new(Int).Lsh(intOne, size-1)
	x := new(Int).Rand(r, n)
	return x.Add(x, n) // make sure result > 1<<(size-1)
}

func runGCD(b *testing.B, aSize, bSize uint) {
	if isRaceBuilder && (aSize > 1000 || bSize > 1000) {
		b.Skip("skipping on race builder")
	}
	b.Run("WithoutXY", func { b -> runGCDExt(b, aSize, bSize, false) })
	b.Run("WithXY", func { b -> runGCDExt(b, aSize, bSize, true) })
}

func runGCDExt(b *testing.B, aSize, bSize uint, calcXY bool) {
	b.StopTimer()
	var r = rand.New(rand.NewSource(1234))
	aa := randInt(r, aSize)
	bb := randInt(r, bSize)
	var x, y *Int
	if calcXY {
		x = new(Int)
		y = new(Int)
	}
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		new(Int).GCD(x, y, aa, bb)
	}
}

func BenchmarkGCD10x10(b *testing.B)         { runGCD(b, 10, 10) }
func BenchmarkGCD10x100(b *testing.B)        { runGCD(b, 10, 100) }
func BenchmarkGCD10x1000(b *testing.B)       { runGCD(b, 10, 1000) }
func BenchmarkGCD10x10000(b *testing.B)      { runGCD(b, 10, 10000) }
func BenchmarkGCD10x100000(b *testing.B)     { runGCD(b, 10, 100000) }
func BenchmarkGCD100x100(b *testing.B)       { runGCD(b, 100, 100) }
func BenchmarkGCD100x1000(b *testing.B)      { runGCD(b, 100, 1000) }
func BenchmarkGCD100x10000(b *testing.B)     { runGCD(b, 100, 10000) }
func BenchmarkGCD100x100000(b *testing.B)    { runGCD(b, 100, 100000) }
func BenchmarkGCD1000x1000(b *testing.B)     { runGCD(b, 1000, 1000) }
func BenchmarkGCD1000x10000(b *testing.B)    { runGCD(b, 1000, 10000) }
func BenchmarkGCD1000x100000(b *testing.B)   { runGCD(b, 1000, 100000) }
func BenchmarkGCD10000x10000(b *testing.B)   { runGCD(b, 10000, 10000) }
func BenchmarkGCD10000x100000(b *testing.B)  { runGCD(b, 10000, 100000) }
func BenchmarkGCD100000x100000(b *testing.B) { runGCD(b, 100000, 100000) }

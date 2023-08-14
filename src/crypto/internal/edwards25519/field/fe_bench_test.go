// Copyright (c) 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package field

import "testing"

func BenchmarkAdd(b *testing.B) {
	x := new(Element).One()
	y := new(Element).Add(x, x)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x.Add(x, y)
	}
}

func BenchmarkMultiply(b *testing.B) {
	x := new(Element).One()
	y := new(Element).Add(x, x)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x.Multiply(x, y)
	}
}

func BenchmarkSquare(b *testing.B) {
	x := new(Element).Add(feOne, feOne)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x.Square(x)
	}
}

func BenchmarkInvert(b *testing.B) {
	x := new(Element).Add(feOne, feOne)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x.Invert(x)
	}
}

func BenchmarkMult32(b *testing.B) {
	x := new(Element).One()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x.Mult32(x, 0xaa42aa42)
	}
}

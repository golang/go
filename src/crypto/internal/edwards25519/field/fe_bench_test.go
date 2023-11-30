// Copyright (c) 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package field

import "testing"

func BenchmarkAdd(b *testing.B) {
	x := new(Element).One()
	y := new(Element).Add(x, x)
	b.ResetTimer()
	for range b.N {
		x.Add(x, y)
	}
}

func BenchmarkMultiply(b *testing.B) {
	x := new(Element).One()
	y := new(Element).Add(x, x)
	b.ResetTimer()
	for range b.N {
		x.Multiply(x, y)
	}
}

func BenchmarkSquare(b *testing.B) {
	x := new(Element).Add(feOne, feOne)
	b.ResetTimer()
	for range b.N {
		x.Square(x)
	}
}

func BenchmarkInvert(b *testing.B) {
	x := new(Element).Add(feOne, feOne)
	b.ResetTimer()
	for range b.N {
		x.Invert(x)
	}
}

func BenchmarkMult32(b *testing.B) {
	x := new(Element).One()
	b.ResetTimer()
	for range b.N {
		x.Mult32(x, 0xaa42aa42)
	}
}

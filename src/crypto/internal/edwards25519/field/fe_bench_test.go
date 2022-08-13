// Copyright (c) 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package field

import "testing"

func BenchmarkAdd(b *testing.B) {
	var x, y Element
	x.One()
	y.Add(feOne, feOne)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x.Add(&x, &y)
	}
}

func BenchmarkMultiply(b *testing.B) {
	var x, y Element
	x.One()
	y.Add(feOne, feOne)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x.Multiply(&x, &y)
	}
}

func BenchmarkMult32(b *testing.B) {
	var x Element
	x.One()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x.Mult32(&x, 0xaa42aa42)
	}
}

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fiat_test

import (
	"crypto/elliptic/internal/fiat"
	"testing"
)

func BenchmarkMul(b *testing.B) {
	b.Run("P224", func(b *testing.B) {
		v := new(fiat.P224Element).One()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			v.Mul(v, v)
		}
	})
	b.Run("P384", func(b *testing.B) {
		v := new(fiat.P384Element).One()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			v.Mul(v, v)
		}
	})
	b.Run("P521", func(b *testing.B) {
		v := new(fiat.P521Element).One()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			v.Mul(v, v)
		}
	})
}

func BenchmarkSquare(b *testing.B) {
	b.Run("P224", func(b *testing.B) {
		v := new(fiat.P224Element).One()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			v.Square(v)
		}
	})
	b.Run("P384", func(b *testing.B) {
		v := new(fiat.P384Element).One()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			v.Square(v)
		}
	})
	b.Run("P521", func(b *testing.B) {
		v := new(fiat.P521Element).One()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			v.Square(v)
		}
	})
}

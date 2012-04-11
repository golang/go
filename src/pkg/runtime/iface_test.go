// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"bytes"
	"io"
	"testing"
)

var (
	I   interface{}
	J   int
	B                 = new(bytes.Buffer)
	W   io.Writer     = B
	I2  interface{}   = B
	R   io.ReadWriter = B
	Big [2]*int
)

func BenchmarkConvT2E(b *testing.B) {
	for i := 0; i < b.N; i++ {
		I = 1
	}
}

func BenchmarkConvT2EBig(b *testing.B) {
	v := [2]*int{}
	for i := 0; i < b.N; i++ {
		I = v
	}
}

func BenchmarkConvT2I(b *testing.B) {
	for i := 0; i < b.N; i++ {
		W = B
	}
}

func BenchmarkConvI2E(b *testing.B) {
	for i := 0; i < b.N; i++ {
		I = W
	}
}

func BenchmarkConvI2I(b *testing.B) {
	for i := 0; i < b.N; i++ {
		W = R
	}
}

func BenchmarkAssertE2T(b *testing.B) {
	I = 1
	for i := 0; i < b.N; i++ {
		J = I.(int)
	}
}

func BenchmarkAssertE2TBig(b *testing.B) {
	var v interface{} = [2]*int{}
	for i := 0; i < b.N; i++ {
		Big = v.([2]*int)
	}
}

func BenchmarkAssertE2I(b *testing.B) {
	for i := 0; i < b.N; i++ {
		W = I2.(io.Writer)
	}
}

func BenchmarkAssertI2T(b *testing.B) {
	for i := 0; i < b.N; i++ {
		B = W.(*bytes.Buffer)
	}
}

func BenchmarkAssertI2I(b *testing.B) {
	for i := 0; i < b.N; i++ {
		W = R.(io.Writer)
	}
}

func BenchmarkAssertI2E(b *testing.B) {
	for i := 0; i < b.N; i++ {
		I = R.(interface{})
	}
}

func BenchmarkAssertE2E(b *testing.B) {
	for i := 0; i < b.N; i++ {
		I = I2.(interface{})
	}
}

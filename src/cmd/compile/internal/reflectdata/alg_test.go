// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflectdata_test

import (
	"testing"
)

func BenchmarkEqArrayOfStrings5(b *testing.B) {
	var a [5]string
	var c [5]string

	for i := 0; i < 5; i++ {
		a[i] = "aaaa"
		c[i] = "cccc"
	}

	for j := 0; j < b.N; j++ {
		_ = a == c
	}
}

func BenchmarkEqArrayOfStrings64(b *testing.B) {
	var a [64]string
	var c [64]string

	for i := 0; i < 64; i++ {
		a[i] = "aaaa"
		c[i] = "cccc"
	}

	for j := 0; j < b.N; j++ {
		_ = a == c
	}
}

func BenchmarkEqArrayOfStrings1024(b *testing.B) {
	var a [1024]string
	var c [1024]string

	for i := 0; i < 1024; i++ {
		a[i] = "aaaa"
		c[i] = "cccc"
	}

	for j := 0; j < b.N; j++ {
		_ = a == c
	}
}

func BenchmarkEqArrayOfFloats5(b *testing.B) {
	var a [5]float32
	var c [5]float32

	for i := 0; i < b.N; i++ {
		_ = a == c
	}
}

func BenchmarkEqArrayOfFloats64(b *testing.B) {
	var a [64]float32
	var c [64]float32

	for i := 0; i < b.N; i++ {
		_ = a == c
	}
}

func BenchmarkEqArrayOfFloats1024(b *testing.B) {
	var a [1024]float32
	var c [1024]float32

	for i := 0; i < b.N; i++ {
		_ = a == c
	}
}

func BenchmarkEqArrayOfStructsEq(b *testing.B) {
	type T2 struct {
		a string
		b int
	}
	const size = 1024
	var (
		str1 = "foobar"

		a [size]T2
		c [size]T2
	)

	for i := 0; i < size; i++ {
		a[i].a = str1
		c[i].a = str1
	}

	b.ResetTimer()
	for j := 0; j < b.N; j++ {
		_ = a == c
	}
}

func BenchmarkEqArrayOfStructsNotEq(b *testing.B) {
	type T2 struct {
		a string
		b int
	}
	const size = 1024
	var (
		str1 = "foobar"
		str2 = "foobarz"

		a [size]T2
		c [size]T2
	)

	for i := 0; i < size; i++ {
		a[i].a = str1
		c[i].a = str1
	}
	c[len(c)-1].a = str2

	b.ResetTimer()
	for j := 0; j < b.N; j++ {
		_ = a == c
	}
}

const size = 16

type T1 struct {
	a [size]byte
}

func BenchmarkEqStruct(b *testing.B) {
	x, y := T1{}, T1{}
	x.a = [size]byte{1, 2, 3, 4, 5, 6, 7, 8}
	y.a = [size]byte{2, 3, 4, 5, 6, 7, 8, 9}

	for i := 0; i < b.N; i++ {
		f := x == y
		if f {
			println("hello")
		}
	}
}

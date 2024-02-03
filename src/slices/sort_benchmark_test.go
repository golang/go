// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slices_test

import (
	"fmt"
	"slices"
	"testing"
)

func BenchmarkBinarySearchFloats(b *testing.B) {
	for _, size := range []int{16, 32, 64, 128, 512, 1024} {
		b.Run(fmt.Sprintf("Size%d", size), func(b *testing.B) {
			floats := make([]float64, size)
			for i := range floats {
				floats[i] = float64(i)
			}
			midpoint := len(floats) / 2
			needle := (floats[midpoint] + floats[midpoint+1]) / 2
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				slices.BinarySearch(floats, needle)
			}
		})
	}
}

type myStruct struct {
	a, b, c, d string
	n          int
}

func BenchmarkBinarySearchFuncStruct(b *testing.B) {
	for _, size := range []int{16, 32, 64, 128, 512, 1024} {
		b.Run(fmt.Sprintf("Size%d", size), func(b *testing.B) {
			structs := make([]*myStruct, size)
			for i := range structs {
				structs[i] = &myStruct{n: i}
			}
			midpoint := len(structs) / 2
			needle := &myStruct{n: (structs[midpoint].n + structs[midpoint+1].n) / 2}
			lessFunc := func(a, b *myStruct) int { return a.n - b.n }
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				slices.BinarySearchFunc(structs, needle, lessFunc)
			}
		})
	}
}

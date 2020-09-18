// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import "testing"

var globl int64

func BenchmarkLoadAdd(b *testing.B) {
	x := make([]int64, 1024)
	y := make([]int64, 1024)
	for i := 0; i < b.N; i++ {
		var s int64
		for i := range x {
			s ^= x[i] + y[i]
		}
		globl = s
	}
}

func BenchmarkModify(b *testing.B) {
	a := make([]int64, 1024)
	v := globl
	for i := 0; i < b.N; i++ {
		for j := range a {
			a[j] += v
		}
	}
}

func BenchmarkConstModify(b *testing.B) {
	a := make([]int64, 1024)
	for i := 0; i < b.N; i++ {
		for j := range a {
			a[j] += 3
		}
	}
}

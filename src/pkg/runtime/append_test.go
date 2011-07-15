// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package runtime_test

import "testing"

const N = 20

func BenchmarkAppend(b *testing.B) {
	b.StopTimer()
	x := make([]int, 0, N)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		x = x[0:0]
		for j := 0; j < N; j++ {
			x = append(x, j)
		}
	}
}

func BenchmarkAppendSpecialCase(b *testing.B) {
	b.StopTimer()
	x := make([]int, 0, N)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		x = x[0:0]
		for j := 0; j < N; j++ {
			if len(x) < cap(x) {
				x = x[:len(x)+1]
				x[len(x)-1] = j
			} else {
				x = append(x, j)
			}
		}
	}
}

var x []int

func f() int {
	x[:1][0] = 3
	return 2
}

func TestSideEffectOrder(t *testing.T) {
	x = make([]int, 0, 10)
	x = append(x, 1, f())
	if x[0] != 1 || x[1] != 2 {
		t.Error("append failed: ", x[0], x[1])
	}
}

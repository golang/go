// compile

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type Float64Slice []float64

func (a Float64Slice) Search1(x float64) int {
	f := func(q int) bool { return a[q] >= x }
	i := 0
	if !f(3) {
		i = 5
	}
	return i
}

// compile

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func F(s string) bool {
	const m = 16
	const n = 1e5
	_ = make([]int, n)
	return len(s) < n*m
}

func G() {
	const n = 1e5
	_ = make([]int, n)
	f := n
	var _ float64 = f
}

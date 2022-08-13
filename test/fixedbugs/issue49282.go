// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

//go:noinline
func g(d uintptr, a, m []int, s struct {
	a, b, c, d, e int
}, u uint) {
	_ = a
	_ = m
	_ = s
	func() {
		for i := 0; i < 5; i++ {
			_ = a
			_ = m
			_, _ = s, s
		}
	}()
}

var One float64 = 1.0

func f(d uintptr) {
	var a, m []int
	var s struct {
		a, b, c, d, e int
	}

	g(d, a, m, s, uint(One)) // Uint of not-a-constant inserts a conditional, necessary to bug

	defer func() uint {
		return 0
	}()
}

var d uintptr

func h() {
	f(d)
}

// compile

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 20097: ensure that we CSE multiple Select ops with
// the same underlying type

package main

type T int64

func f(x, y int64) (int64, T) {
	a := x / y
	b := T(x) / T(y)
	return a, b
}

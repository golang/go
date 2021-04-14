// compile

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var (
	x  int
	xs []int
)

func a([]int) (int, error)

func b() (int, error) {
	return a(append(xs, x))
}

func c(int, error) (int, error)

func d() (int, error) {
	return c(b())
}

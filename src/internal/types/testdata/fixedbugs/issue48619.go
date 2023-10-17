// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f[P any](a, _ P) {
	var x int
	// TODO(gri) these error messages, while correct, could be better
	f(a, x /* ERROR "type int of x does not match inferred type P for P" */)
	f(x, a /* ERROR "type P of a does not match inferred type int for P" */)
}

func g[P any](a, b P) {
	g(a, b)
	g(&a, &b)
	g([]P{}, []P{})

	// work-around: provide type argument explicitly
	g[*P](&a, &b)
	g[[]P]([]P{}, []P{})
}

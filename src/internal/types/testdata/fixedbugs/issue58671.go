// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func g[P any](...P) P { var x P; return x }

func _() {
	var (
		_ int        = g(1, 2)
		_ rune       = g(1, 'a')
		_ float64    = g(1, 'a', 2.3)
		_ float64    = g('a', 2.3)
		_ complex128 = g(2.3, 'a', 1i)
	)
	g(true, 'a' /* ERROR "mismatched types untyped bool and untyped rune (cannot infer P)" */)
	g(1, "foo" /* ERROR "mismatched types untyped int and untyped string (cannot infer P)" */)
	g(1, 2.3, "bar" /* ERROR "mismatched types untyped float and untyped string (cannot infer P)" */)
}

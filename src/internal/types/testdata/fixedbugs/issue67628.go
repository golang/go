// -gotypesalias=1

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f[P any](x P) P { return x }

func _() {
	type A = int
	var a A
	b := f(a) // type of b is A
	// error should report type of b as A, not int
	_ = b /* ERROR "mismatched types A and untyped string" */ + "foo"
}

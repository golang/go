// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that there are no type inference errors
// if function arguments are invalid.

package p

func f[S any](S) {}

var s struct{ x int }

func _() {
	f(s.y /* ERROR "s.y undefined" */)
	f(1 /* ERROR "invalid operation: 1 / s (mismatched types untyped int and struct{x int})" */ / s)
}

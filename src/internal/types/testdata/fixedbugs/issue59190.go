// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "unsafe"

type E [1 << 30]complex128
var a [1 << 30]E
var _ = unsafe.Sizeof(a /* ERROR "too large" */ )

var s struct {
	_ [1 << 30]E
	x int
}
var _ = unsafe.Offsetof(s /* ERROR "too large" */ .x)

// Test case from issue (modified so it also triggers on 32-bit platforms).

type A [1]int
type S struct {
	x A
	y [1 << 30]A
	z [1 << 30]struct{}
}
type T [1 << 30][1 << 30]S

func _() {
	var a A
	var s S
	var t T
	_ = unsafe.Sizeof(a)
	_ = unsafe.Sizeof(s)
	_ = unsafe.Sizeof(t /* ERROR "too large" */ )
}

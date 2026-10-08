// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var s int
var _ int = (1.0 << s) << 1
var _ int = (1.1 /* ERROR "must be integer" */ << s) << 1
var _ = (1.0 /* ERROR "must be integer" */ << s) << 1

// test case from issue

func _(x int, s uint) {
	x = (1. << s) << (1 << s)
	x = (1. << s) << (1. << s)
}

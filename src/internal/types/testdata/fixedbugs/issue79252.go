// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _() {
	var s int
	_ = 1 << (- /* ERROR "overflows uint" */ 1 << s)
	_ = 1 << (1 << 63 << s)
	_ = 1 << (1 /* ERROR "overflows uint" */ << 64 << s)
	_ = 1 << (1 /* ERROR "overflows uint" */ << 65 << s)

	_ = 1 << (1.0 << s)
	_ = 1 << (- /* ERROR "truncated to uint" */ 1.0 << s)
	_ = 1 << (1.1 /* ERROR "must be integer" */ << s)
}

func _() {
	var s int
	var _ float32 = (1 /* ERROR "must be integer" */ << 200) << s
}

// test case from issue

func main() {
	var v7 int
	v8 := 0 >> ((2 /* ERROR "overflows uint" */ ^ 0xffffffffffffffff<<2) >> v7)
	_ = v8
}

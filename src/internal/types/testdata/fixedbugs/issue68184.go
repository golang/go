// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type VeryLongStruct struct {
	A1  int
	A2  int
	A3  int
	A4  int
	A5  int
	A6  int
	A7  int
	A8  int
	A9  int
	A10 int
	A11 int
	A12 int
	A13 int
	A14 int
	A15 int
	A16 int
	A17 int
	A18 int
	A19 int
	A20 int
}

func _() {
	// The error messages in both these cases should print the
	// struct name rather than the struct's underlying type.

	var x VeryLongStruct
	x.B2 /* ERROR "x.B2 undefined (type VeryLongStruct has no field or method B2)" */ = false

	_ = []VeryLongStruct{{B2 /* ERROR "unknown field B2 in struct literal of type VeryLongStruct" */ : false}}
}

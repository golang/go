// compile

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Unified frontend generated unnecessary temporaries for expressions
// within unsafe.Sizeof, etc functions.

package main

import "unsafe"

func F[G int](g G) (uintptr, uintptr, uintptr) {
	var c chan func() int
	type s struct {
		g G
		x []int
	}
	return unsafe.Sizeof(s{g, make([]int, (<-c)())}),
		unsafe.Alignof(s{g, make([]int, (<-c)())}),
		unsafe.Offsetof(s{g, make([]int, (<-c)())}.x)
}

func main() {
	F(0)
}

// errorcheck -0 -live

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T struct { a, b, c, d string } // pass in registers, not SSA-able

//go:registerparams
func F() (r T) {
	r.a = g(1) // ERROR "live at call to g: r"
	r.b = g(2) // ERROR "live at call to g: r"
	r.c = g(3) // ERROR "live at call to g: r"
	r.d = g(4) // ERROR "live at call to g: r"
	return
}

func g(int) string

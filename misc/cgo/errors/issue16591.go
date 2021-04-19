// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 16591: Test that we detect an invalid call that was being
// hidden by a type conversion inserted by cgo checking.

package p

// void f(int** p) { }
import "C"

type x *C.int

func F(p *x) {
	C.f(p) // ERROR HERE
}

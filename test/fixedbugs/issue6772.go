// errorcheck

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f1() {
	for a, a := range []int{1, 2, 3} { // ERROR "a repeated on left side of :="
		println(a)
	}
}

func f2() {
	var a int
	for a, a := range []int{1, 2, 3} { // ERROR "a repeated on left side of :="
		println(a)
	}
	println(a)
}

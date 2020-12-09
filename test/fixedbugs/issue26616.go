// errorcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var x int = three() // ERROR "assignment mismatch: 1 variable but three returns 3 values|3\-valued"

func f() {
	var _ int = three() // ERROR "assignment mismatch: 1 variable but three returns 3 values|3\-valued"
	var a int = three() // ERROR "assignment mismatch: 1 variable but three returns 3 values|3\-valued"
	a = three()         // ERROR "assignment mismatch: 1 variable but three returns 3 values|cannot assign"
	b := three()        // ERROR "assignment mismatch: 1 variable but three returns 3 values|cannot initialize"

	_, _ = a, b
}

func three() (int, int, int)

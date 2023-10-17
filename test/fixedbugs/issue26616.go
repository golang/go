// errorcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var x int = three() // ERROR "assignment mismatch: 1 variable but three returns 3 values|multiple-value function call in single-value context|multiple-value "

func f() {
	var _ int = three() // ERROR "assignment mismatch: 1 variable but three returns 3 values|multiple-value function call in single-value context|multiple-value "
	var a int = three() // ERROR "assignment mismatch: 1 variable but three returns 3 values|multiple-value function call in single-value context|multiple-value "
	a = three()         // ERROR "assignment mismatch: 1 variable but three returns 3 values|multiple-value function call in single-value context|cannot assign"
	b := three()        // ERROR "assignment mismatch: 1 variable but three returns 3 values|single variable set to multiple-value|multiple-value function call in single-value context|cannot initialize"
	_, _ = a, b
}

func three() (int, int, int)

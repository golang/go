// errorcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var a = twoResults()       // ERROR "assignment mismatch: 1 variable but twoResults returns 2 values|multiple-value twoResults\(\) .*in single-value context"
var b, c, d = twoResults() // ERROR "assignment mismatch: 3 variables but twoResults returns 2 values|cannot initialize"
var e, f = oneResult()     // ERROR "assignment mismatch: 2 variables but oneResult returns 1 value|cannot initialize"

func twoResults() (int, int) {
	return 1, 2
}

func oneResult() int {
	return 1
}

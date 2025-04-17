// errorcheck

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var c, d = 1, 2, 3 // ERROR "assignment mismatch: 2 variables but 3 values|wrong number of initializations|extra init expr"
	var e, f, g = 1, 2 // ERROR "assignment mismatch: 3 variables but 2 values|wrong number of initializations|missing init expr"
	_, _, _, _, _ = c, d, e, f, g
}

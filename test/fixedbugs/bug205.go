// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var t []int
var s string;
var m map[string]int;

func main() {
	println(t["hi"]); // ERROR "non-integer slice index|must be integer|cannot convert"
	println(s["hi"]); // ERROR "non-integer string index|must be integer|cannot convert"
	println(m[0]);    // ERROR "cannot use.*as type string|cannot convert|cannot use"
}


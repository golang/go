// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 19012: if we have any unknown type at a call site,
// we must ensure that we return to the user a suppressed
// error message saying instead of including <T> in
// the message.

package main

func f(x int, y uint) {
	if true {
		return "a" > 10 // ERROR "^too many arguments to return$" "."
	}
	return "gopher" == true, 10 // ERROR "^too many arguments to return$" "."
}

func main() {
	f(2, 3 < "x", 10) // ERROR "^too many arguments in call to f$" "."

	f(10, 10, "a") // ERROR "too many arguments in call to f\n\thave \(number, number, string\)\n\twant \(int, uint\)"
}

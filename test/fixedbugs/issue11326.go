// errorcheck

// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

func main() {
	var g = 1e81391777742999 // ERROR "exponent too large"
	// The next should only cause a problem when converted to float64
	// by the assignment, but instead the compiler rejects it outright,
	// rather than mishandle it. Specifically, when handled, 'var h' prints:
	//	issue11326.go:N: constant 0.93342e+536870911 overflows float64
	// The rejection of 'var i' is just insurance. It seems to work correctly.
	// See golang.org/issue/11326.
	// var h = 1e2147483647     // should be "1.00000e+2147483647 overflows float64"
	var h = 1e2147483647 // ERROR "exponent too large"
	// var i = 1e214748364  // should be "1.00000e\+214748364 overflows float64"
	var i = 1e214748364 // ERROR "exponent too large"
	var j = 1e21474836  // ERROR "1.00000e\+21474836 overflows float64"
	var k = 1e2147483   // ERROR "1.00000e\+2147483 overflows float64"
	var l = 1e214748    // ERROR "1.00000e\+214748 overflows float64"
	var m = 1e21474     // ERROR "1.00000e\+21474 overflows float64"
	fmt.Println(g)
}

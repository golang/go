// errorcheckwithauto -0 -m -d=inlfuncswithclosures=1
//go:build goexperiment.unified
// +build goexperiment.unified

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package foo

func r(z int) int {
	foo := func(x int) int { // ERROR "can inline r.func1" "func literal does not escape"
		return x + z
	}
	bar := func(x int) int { // ERROR "func literal does not escape" "can inline r.func2"
		return x + func(y int) int { // ERROR "can inline r.func2.1" "can inline r.func3"
			return 2*y + x*z
		}(x) // ERROR "inlining call to r.func2.1"
	}
	return foo(42) + bar(42) // ERROR "inlining call to r.func1" "inlining call to r.func2" "inlining call to r.func3"
}

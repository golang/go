// errorcheck -0 -m

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis for functions with variadic arguments

package foo

func debugf(format string, args ...interface{}) { // ERROR "can inline debugf" "format does not escape" "args does not escape"
	// Dummy implementation for non-debug build.
	// A non-empty implementation would be enabled with a build tag.
}

func bar() { // ERROR "can inline bar"
	value := 10
	debugf("value is %d", value) // ERROR "inlining call to debugf" "value does not escape" "\[\]interface {} literal does not escape"
}

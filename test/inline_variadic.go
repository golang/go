// errorcheck -0 -m

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test inlining of variadic functions.
// See issue #18116.

package foo

func head(xs ...string) string { // ERROR "can inline head" "leaking param: xs to result"
	return xs[0]
}

func f() string { // ERROR "can inline f"
	x := head("hello", "world") // ERROR "inlining call to head" "\[\]string literal does not escape"
	return x
}

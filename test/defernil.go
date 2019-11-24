// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check that deferring a nil function causes a proper
// panic when the deferred function is invoked (not
// when the function is deferred).
// See Issue #8047 and #34926.

package main

var x = 0

func main() {
	defer func() {
		err := recover()
		if err == nil {
			panic("did not panic")
		}
		if x != 1 {
			panic("FAIL")
		}
	}()
	f()
}

func f() {
	var nilf func()
	defer nilf()
	x = 1
}

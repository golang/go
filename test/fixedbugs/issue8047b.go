// run

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 8047. Defer setup during panic shouldn't crash for nil defer.

package main

func main() {
	defer func() {
		// This recover recovers the panic caused by the nil defer func
		// g(). The original panic(1) was already aborted/replaced by this
		// new panic, so when this recover is done, the program completes
		// normally.
		recover()
	}()
	f()
}

func f() {
	var g func()
	defer g()
	panic(1)
}

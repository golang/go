// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The inliner would erroneously scan the caller function's body for
// reassignments *before* substituting the inlined function call body,
// which could cause false positives in deciding when it's safe to
// transitively inline indirect function calls.

package main

func main() {
	bug1()
	bug2(fail)
}

func bug1() {
	fn := fail
	fn = pass
	fn()
}

func bug2(fn func()) {
	fn = pass
	fn()
}

func pass() {}
func fail() { panic("FAIL") }

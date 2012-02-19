// run

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test closures in if conditions.

package main

func main() {
	if func() bool { return true }() {}  // 6g used to say this was a syntax error
	if (func() bool { return true })() {}
	if (func() bool { return true }()) {}
}


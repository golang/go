// run

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 8047. Defer setup during panic shouldn't crash for nil defer.

package main

func main() {
	defer func() {
		recover()
	}()
	f()
}

func f() {
	var g func()
	defer g()
	panic(1)
}

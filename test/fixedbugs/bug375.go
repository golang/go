// run

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 2423

package main

func main() {
	var x interface{} = "hello"

	switch x {
	case "hello":
	default:
		println("FAIL")
	}
}

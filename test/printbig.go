// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that big numbers work as constants and print can print them.

package main

func main() {
	print(-(1<<63), "\n")
	print((1<<63)-1, "\n")
}

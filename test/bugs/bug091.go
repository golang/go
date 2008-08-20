// errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f1() {
	exit:
		print("hi\n");
}

func f2() {
	const c = 1234;
}

func f3() {
	i := c;	// BUG: compiles but should not. constant is not in scope in this function
	goto exit;	// BUG: compiles but should not. label is not in this function
}

func main() {
	f3();
}

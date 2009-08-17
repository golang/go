// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	c := 10;
	d := 7;
	var x [10]int;
	i := 0;
	/* this works:
	q := c/d;
	x[i] = q;
	*/
	// this doesn't:
	x[i] = c/d;	// BUG segmentation fault
}

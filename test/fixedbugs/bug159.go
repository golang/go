// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "os"

func main() {
	ok := true;
	var a, b, c, x, y, z int;
	f := func() int { b--; return -b };

	// this fails on 6g: apparently it rewrites
	// the list into
	//	z = f();
	//	y = f();
	//	x = f();
	// so that the values come out backward.
	x, y, z = f(), f(), f();
	if x != 1 || y != 2 || z != 3 {
		println("xyz: expected 1 2 3 got", x, y, z);
		ok = false;
	}

	// this fails on 6g too.  one of the function calls
	// happens after assigning to b.
	a, b, c = f(), f(), f();
	if a != 4 || b != 5 || c != 6 {
		println("abc: expected 4 5 6 got", a, b, c);
		ok = false;
	}

	if !ok {
		os.Exit(1);
	}
}

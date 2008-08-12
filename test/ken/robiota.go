// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// $G $D/$F.go && $L $F.$A && ./$A.out

package main

func assert(cond bool, msg string) {
	if !cond {
		print("assertion fail: " + msg + "\n");
		panic(1);
	}
}

const (
	x int = iota;
	y = iota;
	z = 1 << iota;
	f float = 2 * iota;
	g float = 4.5 * float(iota);
);

func main() {
	assert(x == 0, "x");
	assert(y == 1, "y");
	assert(z == 4, "z");
	assert(f == 6.0, "f");
	assert(g == 18.0, "g");
}

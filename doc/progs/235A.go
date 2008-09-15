// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type INT uint64

func Multiplier(f INT, in, out *chan INT) {
	for {
		out -< (f * <-in);
	}
}

func min(a, b INT) INT {
	if a < b { return a }
	return b;
}

func main() {
	c2i := new(chan INT, 100);
	c2o := new(chan INT);
	c3i := new(chan INT, 100);
	c3o := new(chan INT);
	c5i := new(chan INT, 100);
	c5o := new(chan INT);

	go Multiplier(2, c2i, c2o);
	go Multiplier(3, c3i, c3o);
	go Multiplier(5, c5i, c5o);

	var x INT = 1;

	x2 := x;
	x3 := x;
	x5 := x;

	for i := 0; i < 100; i++ {
		print(x, "\n");

		c2i -< x;
		c3i -< x;
		c5i -< x;

		if x2 == x { x2 = <- c2o }
		if x3 == x { x3 = <- c3o }
		if x5 == x { x5 = <- c5o }

		x = min(min(x2, x3), x5);
	}
	sys.exit(0);
}

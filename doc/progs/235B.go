// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type INT uint64

func Multiplier(f INT) (in, out *chan INT) {
	inc := new(chan INT, 100);
	outc := new(chan INT);
	go func(f INT, in, out *chan INT) {
		for {
			out -< f * <-in;
		}
	}(f, inc, outc)
	return inc, outc
}

func min(a, b INT) INT {
	if a < b { return a }
	return b;
}

func main() {
	c2i, c2o := Multiplier(2);
	c3i, c3o := Multiplier(3);
	c5i, c5o := Multiplier(5);

	var x INT = 1;

	x2, x3, x5 := x, x, x;

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

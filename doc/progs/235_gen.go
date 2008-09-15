// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type INT uint64

func Multiplier(f INT) (in, out *chan INT) {
	in = new(chan INT, 100);
	out = new(chan INT, 100);
	go func(in, out *chan INT, f INT) {
		for {
			out -< f * <- in;
		}
	}(in, out, f);
	return in, out;
}


func min(xs *[]INT) INT {
	m := xs[0];
	for i := 1; i < len(xs); i++ {
		if xs[i] < m {
			m = xs[i];
		}
	}
	return m;
}


func main() {
	F := []INT{2, 3, 5};
	const n = len(F);

	x := INT(1);
	ins := new([]*chan INT, n);
	outs := new([]*chan INT, n);
	xs := new([]INT, n);
	for i := 0; i < n; i++ {
		ins[i], outs[i] = Multiplier(F[i]);
		xs[i] = x;
	}

	for i := 0; i < 100; i++ {
		print(x, "\n");
		t := min(xs);
		for i := 0; i < n; i++ {
			ins[i] -< x;
		}

		for i := 0; i < n; i++ {
			if xs[i] == x { xs[i] = <- outs[i]; }
		}
		
		x = min(xs);
	}
	sys.exit(0);
}

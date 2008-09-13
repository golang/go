// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T chan uint64;

func M(f uint64) (in, out *T) {
	in = new(T, 100);
	out = new(T, 100);
	go func(in, out *T, f uint64) {
		for {
			out -< f * <- in;
		}
	}(in, out, f);
	return in, out;
}


func min(xs *[]uint64) uint64 {
	m := xs[0];
	for i := 1; i < len(xs); i++ {
		if xs[i] < m {
			m = xs[i];
		}
	}
	return m;
}


func main() {
	F := []uint64{2, 3, 5};
	const n = len(F);

	x := uint64(1);
	ins := new([]*T, n);
	outs := new([]*T, n);
	xs := new([]uint64, n);
	for i := 0; i < n; i++ {
		ins[i], outs[i] = M(F[i]);
		xs[i] = x;
	}

	for i := 0; i < 100; i++ {
		t := min(xs);
		for i := 0; i < n; i++ {
			ins[i] -< x;
		}

		for i := 0; i < n; i++ {
			if xs[i] == x { xs[i] = <- outs[i]; }
		}
		
		x = min(xs);
		print(x, "\n");
	}
	sys.exit(0);
}

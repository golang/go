// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func assert(cond bool, msg string) {
	if !cond {
		print("assertion fail: ", msg, "\n");
		panic(1);
	}
}

const (
	x int = iota;
	y = iota;
	z = 1 << iota;
	f float = 2 * iota;
	g float = 4.5 * float(iota);
)

const (
	X = 0;
	Y;
	Z;
)

const (
	A = 1 << iota;
	B;
	C;
	D;
	E = iota * iota;
	F;
	G;
)

const (
	a = 1;
	b = iota << a;
	c = iota << b;
	d;
)

const (
	i = (a << iota) + (b * iota);
	j;
	k;
	l;
)

const (
	m = iota == 0;
	n;
)

const (
	p = float(iota);
	q;
	r;
)

const (
	s = string(iota + 'a');
	t;
)

func main() {
	assert(x == 0, "x");
	assert(y == 1, "y");
	assert(z == 4, "z");
	assert(f == 6.0, "f");
	assert(g == 18.0, "g");

	assert(X == 0, "X");
	assert(Y == 0, "Y");
	assert(Z == 0, "Z");

	assert(A == 1, "A");
	assert(B == 2, "B");
	assert(C == 4, "C");
	assert(D == 8, "D");
	assert(E == 16, "E");
	assert(F == 25, "F");

	assert(a == 1, "a");
	assert(b == 2, "b");
	assert(c == 8, "c");
	assert(d == 12, "d");

	assert(i == 1, "i");
	assert(j == 4, "j");
	assert(k == 8, "k");
	assert(l == 14, "l");

	assert(m, "m");
	assert(!n, "n");

	assert(p == 0.0, "p");
	assert(q == 1.0, "q");
	assert(r == 2.0, "r");

	assert(s == "a", "s");
	assert(t == "b", "t");
}

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package expressions

type T struct {
	x, y, z int
}

var (
	a, b, c, d, e int;
	longIdentifier1, longIdentifier2, longIdentifier3 int;
	t0, t1, t2 T;
	s string;
)

func main() {
	// no spaces around simple or parenthesized expressions
	_ = a+b;
	_ = a+b+c;
	_ = a+b-c;
	_ = a-b-c;
	_ = a+(b*c);
	_ = a+(b/c);
	_ = a-(b%c);
	_ = 1+a;
	_ = a+1;
	_ = a+b+1;
	_ = "foo"+s;
	_ = s+"foo";
	_ = s[1:2];
	_ = s[a:b];
	_ = s[0:len(s)];

	// spaces around expressions of different precedence or expressions containing spaces
	_ = a + -b;
	_ = a - ^b;
	_ = a / *b;
	_ = a + b*c;
	_ = 1 + b*c;
	_ = a + 2*c;
	_ = a + c*2;
	_ = 1 + 2*3;
	_ = s[1 : 2*3];
	_ = s[a : b-c];
	_ = s[a+b : len(s)];
	_ = s[len(s) : -a];
	_ = s[a : len(s)+1];

	// spaces around operators with equal or lower precedence than comparisons
	_ = a == b;
	_ = a != b;
	_ = a > b;
	_ = a >= b;
	_ = a < b;
	_ = a <= b;
	_ = a < b && c > d;
	_ = a < b || c > d;

	// spaces around "long" operands
	_ = a + longIdentifier1;
	_ = longIdentifier1 + a;
	_ = longIdentifier1 + longIdentifier2 * longIdentifier3;
	_ = s + "a longer string";

	// some selected cases
	_ = a + t0.x;
	_ = a + t0.x + t1.x * t2.x;
	_ = a + b + c + d + e + 2*3;
	_ = a + b + c + 2*3 + d + e;
	_ = (a+b+c)*2;
	_ = a - b + c - d + (a+b+c) + d&e;
}

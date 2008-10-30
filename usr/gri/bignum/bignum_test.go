// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import Bignum "bignum"

const (
	sa = "991";
	sb = "2432902008176640000";  // 20!
	sc = "93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000";  // 100!
)


var (
	a = Bignum.NatFromString(sa, 10);
	b = Bignum.NatFromString(sb, 10);
	c = Bignum.NatFromString(sc, 10);
)


var test_msg string;
func TEST(n int, b bool) {
	if !b {
		panic("TEST failed: ", test_msg, "(", n, ")\n");
	}
}


func TestConv() {
	test_msg = "TestConv";
	TEST(0, a.Cmp(Bignum.NewNat(991)) == 0);
	TEST(1, b.Cmp(Bignum.Fact(20)) == 0);
	TEST(2, c.Cmp(Bignum.Fact(100)) == 0);
	TEST(3, a.String(10) == sa);
	TEST(4, b.String(10) == sb);
	TEST(5, c.String(10) == sc);
}


func TestShift() {
	test_msg = "TestShift1L";
	TEST(0, b.Shl(0).Cmp(b) == 0);
	TEST(1, c.Shl(1).Cmp(c) > 0);
	
	test_msg = "TestShift1R";
	TEST(0, b.Shr(0).Cmp(b) == 0);
	TEST(1, c.Shr(1).Cmp(c) < 0);

	test_msg = "TestShift2";
	for i := 0; i < 100; i++ {
		TEST(i, c.Shl(uint(i)).Shr(uint(i)).Cmp(c) == 0);
	}

	test_msg = "TestShift3L";
	{	const m = 3;
		p := b;
		f := Bignum.NewNat(1<<m);
		for i := 0; i < 100; i++ {
			TEST(i, b.Shl(uint(i*m)).Cmp(p) == 0);
			p = p.Mul(f);
		}
	}

	test_msg = "TestShift3R";
	{	p := c;
		for i := 0; c.Cmp(Bignum.NatZero) == 0; i++ {
			TEST(i, c.Shr(uint(i)).Cmp(p) == 0);
			p = p.Shr(1);
		}
	}
}


func main() {
	TestConv();
	TestShift();
	print("PASSED\n");
}

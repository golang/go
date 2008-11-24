// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bignum_test

import (
	bignum "bignum";
	fmt "fmt";
	testing "testing";
)

const (
	sa = "991";
	sb = "2432902008176640000";  // 20!
	sc = "933262154439441526816992388562667004907159682643816214685929"
	     "638952175999932299156089414639761565182862536979208272237582"
		 "51185210916864000000000000000000000000";  // 100!
	sp = "170141183460469231731687303715884105727";  // prime
)

func NatFromString(s string, base uint, slen *int) *bignum.Natural {
	x, dummy := bignum.NatFromString(s, base, slen);
	return x;
}


func IntFromString(s string, base uint, slen *int) *bignum.Integer {
	x, dummy := bignum.IntFromString(s, base, slen);
	return x;
}


func RatFromString(s string, base uint, slen *int) *bignum.Rational {
	x, dummy := bignum.RatFromString(s, base, slen);
	return x;
}


var (
	nat_zero = bignum.Nat(0);
	nat_one = bignum.Nat(1);
	nat_two = bignum.Nat(2);

	a = NatFromString(sa, 10, nil);
	b = NatFromString(sb, 10, nil);
	c = NatFromString(sc, 10, nil);
	p = NatFromString(sp, 10, nil);

	int_zero = bignum.Int(0);
	int_one = bignum.Int(1);
	int_two = bignum.Int(2);

	ip = IntFromString(sp, 10, nil);

	rat_zero = bignum.Rat(0, 1);
	rat_half = bignum.Rat(1, 2);
	rat_one = bignum.Rat(1, 1);
	rat_two = bignum.Rat(2, 1);
)


var test_msg string;
var tester *testing.T;

func TEST(n uint, b bool) {
	if !b {
		tester.Fatalf("TEST failed: %s (%d)", test_msg, n);
	}
}


func NAT_EQ(n uint, x, y *bignum.Natural) {
	if x.Cmp(y) != 0 {
		tester.Fatalf("TEST failed: %s (%d)\nx = %v\ny = %v", test_msg, n, x, y);
	}
}


func INT_EQ(n uint, x, y *bignum.Integer) {
	if x.Cmp(y) != 0 {
		tester.Fatalf("TEST failed: %s (%d)\nx = %v\ny = %v", test_msg, n, x, y);
	}
}


func RAT_EQ(n uint, x, y *bignum.Rational) {
	if x.Cmp(y) != 0 {
		tester.Fatalf("TEST failed: %s (%d)\nx = %v\ny = %v", test_msg, n, x, y);
	}
}

export func TestNatConv(t *testing.T) {
	tester = t;
	test_msg = "NatConvA";
	NAT_EQ(0, a, bignum.Nat(991));
	NAT_EQ(1, b, bignum.Fact(20));
	NAT_EQ(2, c, bignum.Fact(100));
	TEST(3, a.String() == sa);
	TEST(4, b.String() == sb);
	TEST(5, c.String() == sc);

	test_msg = "NatConvB";
	var slen int;
	NAT_EQ(0, NatFromString("0", 0, nil), nat_zero);
	NAT_EQ(1, NatFromString("123", 0, nil), bignum.Nat(123));
	NAT_EQ(2, NatFromString("077", 0, nil), bignum.Nat(7*8 + 7));
	NAT_EQ(3, NatFromString("0x1f", 0, nil), bignum.Nat(1*16 + 15));
	NAT_EQ(4, NatFromString("0x1fg", 0, &slen), bignum.Nat(1*16 + 15));
	TEST(4, slen == 4);

	test_msg = "NatConvC";
	tmp := c.Mul(c);
	for base := uint(2); base <= 16; base++ {
		NAT_EQ(base, NatFromString(tmp.ToString(base), base, nil), tmp);
	}

	test_msg = "NatConvD";
	x := bignum.Nat(100);
	y, b := bignum.NatFromString(fmt.sprintf("%b", x), 2, nil);
	NAT_EQ(0, y, x);
}


export func TestIntConv(t *testing.T) {
	tester = t;
	test_msg = "IntConv";
	var slen int;
	INT_EQ(0, IntFromString("0", 0, nil), int_zero);
	INT_EQ(1, IntFromString("-0", 0, nil), int_zero);
	INT_EQ(2, IntFromString("123", 0, nil), bignum.Int(123));
	INT_EQ(3, IntFromString("-123", 0, nil), bignum.Int(-123));
	INT_EQ(4, IntFromString("077", 0, nil), bignum.Int(7*8 + 7));
	INT_EQ(5, IntFromString("-077", 0, nil), bignum.Int(-(7*8 + 7)));
	INT_EQ(6, IntFromString("0x1f", 0, nil), bignum.Int(1*16 + 15));
	INT_EQ(7, IntFromString("-0x1f", 0, nil), bignum.Int(-(1*16 + 15)));
	INT_EQ(8, IntFromString("0x1fg", 0, &slen), bignum.Int(1*16 + 15));
	INT_EQ(9, IntFromString("-0x1fg", 0, &slen), bignum.Int(-(1*16 + 15)));
	TEST(10, slen == 5);
}


export func TestRatConv(t *testing.T) {
	tester = t;
	test_msg = "RatConv";
	var slen int;
	RAT_EQ(0, RatFromString("0", 0, nil), rat_zero);
	RAT_EQ(1, RatFromString("0/1", 0, nil), rat_zero);
	RAT_EQ(2, RatFromString("0/01", 0, nil), rat_zero);
	RAT_EQ(3, RatFromString("0x14/10", 0, &slen), rat_two);
	TEST(4, slen == 7);
	RAT_EQ(5, RatFromString("0.", 0, nil), rat_zero);
	RAT_EQ(6, RatFromString("0.001f", 10, nil), bignum.Rat(1, 1000));
	RAT_EQ(7, RatFromString("10101.0101", 2, nil), bignum.Rat(0x155, 1<<4));
	RAT_EQ(8, RatFromString("-0003.145926", 10, &slen), bignum.Rat(-3145926, 1000000));
	TEST(9, slen == 12);
}


func Add(x, y *bignum.Natural) *bignum.Natural {
	z1 := x.Add(y);
	z2 := y.Add(x);
	if z1.Cmp(z2) != 0 {
		tester.Fatalf("addition not symmetric:\n\tx = %v\n\ty = %t", x, y);
	}
	return z1;
}


func Sum(n uint, scale *bignum.Natural) *bignum.Natural {
	s := nat_zero;
	for ; n > 0; n-- {
		s = Add(s, bignum.Nat(n).Mul(scale));
	}
	return s;
}


export func TestNatAdd(t *testing.T) {
	tester = t;
	test_msg = "NatAddA";
	NAT_EQ(0, Add(nat_zero, nat_zero), nat_zero);
	NAT_EQ(1, Add(nat_zero, c), c);

	test_msg = "NatAddB";
	for i := uint(0); i < 100; i++ {
		t := bignum.Nat(i);
		NAT_EQ(i, Sum(i, c), t.Mul(t).Add(t).Shr(1).Mul(c));
	}
}


func Mul(x, y *bignum.Natural) *bignum.Natural {
	z1 := x.Mul(y);
	z2 := y.Mul(x);
	if z1.Cmp(z2) != 0 {
		tester.Fatalf("multiplication not symmetric:\n\tx = %v\n\ty = %t", x, y);
	}
	if !x.IsZero() && z1.Div(x).Cmp(y) != 0 {
		tester.Fatalf("multiplication/division not inverse (A):\n\tx = %v\n\ty = %t", x, y);
	}
	if !y.IsZero() && z1.Div(y).Cmp(x) != 0 {
		tester.Fatalf("multiplication/division not inverse (B):\n\tx = %v\n\ty = %t", x, y);
	}
	return z1;
}


export func TestNatSub(t *testing.T) {
	tester = t;
	test_msg = "NatSubA";
	NAT_EQ(0, nat_zero.Sub(nat_zero), nat_zero);
	NAT_EQ(1, c.Sub(nat_zero), c);

	test_msg = "NatSubB";
	for i := uint(0); i < 100; i++ {
		t := Sum(i, c);
		for j := uint(0); j <= i; j++ {
			t = t.Sub(Mul(bignum.Nat(j), c));
		}
		NAT_EQ(i, t, nat_zero);
	}
}


export func TestNatMul(t *testing.T) {
	tester = t;
	test_msg = "NatMulA";
	NAT_EQ(0, Mul(c, nat_zero), nat_zero);
	NAT_EQ(1, Mul(c, nat_one), c);

	test_msg = "NatMulB";
	NAT_EQ(0, b.Mul(bignum.MulRange(0, 100)), nat_zero);
	NAT_EQ(1, b.Mul(bignum.MulRange(21, 100)), c);

	test_msg = "NatMulC";
	const n = 100;
	p := b.Mul(c).Shl(n);
	for i := uint(0); i < n; i++ {
		NAT_EQ(i, Mul(b.Shl(i), c.Shl(n-i)), p);
	}
}


export func TestNatDiv(t *testing.T) {
	tester = t;
	test_msg = "NatDivA";
	NAT_EQ(0, c.Div(nat_one), c);
	NAT_EQ(1, c.Div(bignum.Nat(100)), bignum.Fact(99));
	NAT_EQ(2, b.Div(c), nat_zero);
	NAT_EQ(4, nat_one.Shl(100).Div(nat_one.Shl(90)), nat_one.Shl(10));
	NAT_EQ(5, c.Div(b), bignum.MulRange(21, 100));

	test_msg = "NatDivB";
	const n = 100;
	p := bignum.Fact(n);
	for i := uint(0); i < n; i++ {
		NAT_EQ(i, p.Div(bignum.MulRange(1, i)), bignum.MulRange(i+1, n));
	}
}


export func TestIntQuoRem(t *testing.T) {
	tester = t;
	test_msg = "IntQuoRem";
	type T struct { x, y, q, r int };
	a := []T{
		T{+8, +3, +2, +2},
		T{+8, -3, -2, +2},
		T{-8, +3, -2, -2},
		T{-8, -3, +2, -2},
		T{+1, +2,  0, +1},
		T{+1, -2,  0, +1},
		T{-1, +2,  0, -1},
		T{-1, -2,  0, -1},
	};
	for i := uint(0); i < len(a); i++ {
		e := &a[i];
		x, y := bignum.Int(e.x).Mul(ip), bignum.Int(e.y).Mul(ip);
		q, r := bignum.Int(e.q), bignum.Int(e.r).Mul(ip);
		qq, rr := x.QuoRem(y);
		INT_EQ(4*i+0, x.Quo(y), q);
		INT_EQ(4*i+1, x.Rem(y), r);
		INT_EQ(4*i+2, qq, q);
		INT_EQ(4*i+3, rr, r);
	}
}


export func TestIntDivMod(t *testing.T) {
	tester = t;
	test_msg = "IntDivMod";
	type T struct { x, y, q, r int };
	a := []T{
		T{+8, +3, +2, +2},
		T{+8, -3, -2, +2},
		T{-8, +3, -3, +1},
		T{-8, -3, +3, +1},
		T{+1, +2,  0, +1},
		T{+1, -2,  0, +1},
		T{-1, +2, -1, +1},
		T{-1, -2, +1, +1},
	};
	for i := uint(0); i < len(a); i++ {
		e := &a[i];
		x, y := bignum.Int(e.x).Mul(ip), bignum.Int(e.y).Mul(ip);
		q, r := bignum.Int(e.q), bignum.Int(e.r).Mul(ip);
		qq, rr := x.DivMod(y);
		INT_EQ(4*i+0, x.Div(y), q);
		INT_EQ(4*i+1, x.Mod(y), r);
		INT_EQ(4*i+2, qq, q);
		INT_EQ(4*i+3, rr, r);
	}
}


export func TestNatMod(t *testing.T) {
	tester = t;
	test_msg = "NatModA";
	for i := uint(0); ; i++ {
		d := nat_one.Shl(i);
		if d.Cmp(c) < 0 {
			NAT_EQ(i, c.Add(d).Mod(c), d);
		} else {
			NAT_EQ(i, c.Add(d).Div(c), nat_two);
			NAT_EQ(i, c.Add(d).Mod(c), d.Sub(c));
			break;
		}
	}
}


export func TestNatShift(t *testing.T) {
	tester = t;
	test_msg = "NatShift1L";
	TEST(0, b.Shl(0).Cmp(b) == 0);
	TEST(1, c.Shl(1).Cmp(c) > 0);

	test_msg = "NatShift1R";
	TEST(0, b.Shr(0).Cmp(b) == 0);
	TEST(1, c.Shr(1).Cmp(c) < 0);

	test_msg = "NatShift2";
	for i := uint(0); i < 100; i++ {
		TEST(i, c.Shl(i).Shr(i).Cmp(c) == 0);
	}

	test_msg = "NatShift3L";
	{	const m = 3;
		p := b;
		f := bignum.Nat(1<<m);
		for i := uint(0); i < 100; i++ {
			NAT_EQ(i, b.Shl(i*m), p);
			p = Mul(p, f);
		}
	}

	test_msg = "NatShift3R";
	{	p := c;
		for i := uint(0); !p.IsZero(); i++ {
			NAT_EQ(i, c.Shr(i), p);
			p = p.Shr(1);
		}
	}
}


export func TestIntShift(t *testing.T) {
	tester = t;
	test_msg = "IntShift1L";
	TEST(0, ip.Shl(0).Cmp(ip) == 0);
	TEST(1, ip.Shl(1).Cmp(ip) > 0);

	test_msg = "IntShift1R";
	TEST(0, ip.Shr(0).Cmp(ip) == 0);
	TEST(1, ip.Shr(1).Cmp(ip) < 0);

	test_msg = "IntShift2";
	for i := uint(0); i < 100; i++ {
		TEST(i, ip.Shl(i).Shr(i).Cmp(ip) == 0);
	}

	test_msg = "IntShift3L";
	{	const m = 3;
		p := ip;
		f := bignum.Int(1<<m);
		for i := uint(0); i < 100; i++ {
			INT_EQ(i, ip.Shl(i*m), p);
			p = p.Mul(f);
		}
	}

	test_msg = "IntShift3R";
	{	p := ip;
		for i := uint(0); p.IsPos(); i++ {
			INT_EQ(i, ip.Shr(i), p);
			p = p.Shr(1);
		}
	}

	test_msg = "IntShift4R";
	//INT_EQ(0, bignum.Int(-43).Shr(1), bignum.Int(-43 >> 1));
	//INT_EQ(1, ip.Neg().Shr(10), ip.Neg().Div(bignum.Int(1).Shl(10)));
}


export func TestNatCmp(t *testing.T) {
	tester = t;
	test_msg = "NatCmp";
	TEST(0, a.Cmp(a) == 0);
	TEST(1, a.Cmp(b) < 0);
	TEST(2, b.Cmp(a) > 0);
	TEST(3, a.Cmp(c) < 0);
	d := c.Add(b);
	TEST(4, c.Cmp(d) < 0);
	TEST(5, d.Cmp(c) > 0);
}


export func TestNatLog2(t *testing.T) {
	tester = t;
	test_msg = "NatLog2A";
	TEST(0, nat_one.Log2() == 0);
	TEST(1, nat_two.Log2() == 1);
	TEST(2, bignum.Nat(3).Log2() == 1);
	TEST(3, bignum.Nat(4).Log2() == 2);
	
	test_msg = "NatLog2B";
	for i := uint(0); i < 100; i++ {
		TEST(i, nat_one.Shl(i).Log2() == i);
	}
}


export func TestNatGcd(t *testing.T) {
	tester = t;
	test_msg = "NatGcdA";
	f := bignum.Nat(99991);
	NAT_EQ(0, b.Mul(f).Gcd(c.Mul(f)), bignum.MulRange(1, 20).Mul(f));
}


export func TestNatPow(t *testing.T) {
	tester = t;
	test_msg = "NatPowA";
	NAT_EQ(0, nat_two.Pow(0), nat_one);

	test_msg = "NatPowB";
	for i := uint(0); i < 100; i++ {
		NAT_EQ(i, nat_two.Pow(i), nat_one.Shl(i));
	}
}


export func TestNatPop(t *testing.T) {
	tester = t;
	test_msg = "NatPopA";
	TEST(0, nat_zero.Pop() == 0);
	TEST(1, nat_one.Pop() == 1);
	TEST(2, bignum.Nat(10).Pop() == 2);
	TEST(3, bignum.Nat(30).Pop() == 4);
	TEST(4, bignum.Nat(0x1248f).Shl(33).Pop() == 8);

	test_msg = "NatPopB";
	for i := uint(0); i < 100; i++ {
		TEST(i, nat_one.Shl(i).Sub(nat_one).Pop() == i);
	}
}


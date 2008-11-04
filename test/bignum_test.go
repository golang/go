// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// $G $D/$F.go && $L $F.$A && ./$A.out

package main

import Big "bignum"

const (
	sa = "991";
	sb = "2432902008176640000";  // 20!
	sc = "933262154439441526816992388562667004907159682643816214685929"
	     "638952175999932299156089414639761565182862536979208272237582"
		 "51185210916864000000000000000000000000";  // 100!
	sp = "170141183460469231731687303715884105727";  // prime
)


func NatFromString(s string, base uint, slen *int) *Big.Natural {
	x, dummy := Big.NatFromString(s, base, slen);
	return x;
}


func IntFromString(s string, base uint, slen *int) *Big.Integer {
	x, dummy := Big.IntFromString(s, base, slen);
	return x;
}


func RatFromString(s string, base uint, slen *int) *Big.Rational {
	x, dummy := Big.RatFromString(s, base, slen);
	return x;
}


var (
	nat_zero = Big.Nat(0);
	nat_one = Big.Nat(1);
	nat_two = Big.Nat(2);

	a = NatFromString(sa, 10, nil);
	b = NatFromString(sb, 10, nil);
	c = NatFromString(sc, 10, nil);
	p = NatFromString(sp, 10, nil);

	int_zero = Big.Int(0);
	int_one = Big.Int(1);
	int_two = Big.Int(2);

	ip = IntFromString(sp, 10, nil);

	rat_zero = Big.Rat(0, 1);
	rat_half = Big.Rat(1, 2);
	rat_one = Big.Rat(1, 1);
	rat_two = Big.Rat(2, 1);
)


var test_msg string;
func TEST(n uint, b bool) {
	if !b {
		println("TEST failed: ", test_msg, "(", n, ")");
		panic();
	}
}


func NAT_EQ(n uint, x, y *Big.Natural) {
	if x.Cmp(y) != 0 {
		println("TEST failed:", test_msg, "(", n, ")");
		println("x =", x.String(10));
		println("y =", y.String(10));
		panic();
	}
}


func INT_EQ(n uint, x, y *Big.Integer) {
	if x.Cmp(y) != 0 {
		println("TEST failed:", test_msg, "(", n, ")");
		println("x =", x.String(10));
		println("y =", y.String(10));
		panic();
	}
}


func RAT_EQ(n uint, x, y *Big.Rational) {
	if x.Cmp(y) != 0 {
		println("TEST failed:", test_msg, "(", n, ")");
		println("x =", x.String(10));
		println("y =", y.String(10));
		panic();
	}
}


func NatConv() {
	test_msg = "NatConvA";
	NAT_EQ(0, a, Big.Nat(991));
	NAT_EQ(1, b, Big.Fact(20));
	NAT_EQ(2, c, Big.Fact(100));
	TEST(3, a.String(10) == sa);
	TEST(4, b.String(10) == sb);
	TEST(5, c.String(10) == sc);

	test_msg = "NatConvB";
	var slen int;
	NAT_EQ(0, NatFromString("0", 0, nil), nat_zero);
	NAT_EQ(1, NatFromString("123", 0, nil), Big.Nat(123));
	NAT_EQ(2, NatFromString("077", 0, nil), Big.Nat(7*8 + 7));
	NAT_EQ(3, NatFromString("0x1f", 0, nil), Big.Nat(1*16 + 15));
	NAT_EQ(4, NatFromString("0x1fg", 0, &slen), Big.Nat(1*16 + 15));
	TEST(4, slen == 4);

	test_msg = "NatConvC";
	t := c.Mul(c);
	for base := uint(2); base <= 16; base++ {
		NAT_EQ(base, NatFromString(t.String(base), base, nil), t);
	}
}


func IntConv() {
	test_msg = "IntConv";
	var slen int;
	INT_EQ(0, IntFromString("0", 0, nil), int_zero);
	INT_EQ(1, IntFromString("-0", 0, nil), int_zero);
	INT_EQ(2, IntFromString("123", 0, nil), Big.Int(123));
	INT_EQ(3, IntFromString("-123", 0, nil), Big.Int(-123));
	INT_EQ(4, IntFromString("077", 0, nil), Big.Int(7*8 + 7));
	INT_EQ(5, IntFromString("-077", 0, nil), Big.Int(-(7*8 + 7)));
	INT_EQ(6, IntFromString("0x1f", 0, nil), Big.Int(1*16 + 15));
	INT_EQ(7, IntFromString("-0x1f", 0, nil), Big.Int(-(1*16 + 15)));
	INT_EQ(8, IntFromString("0x1fg", 0, &slen), Big.Int(1*16 + 15));
	INT_EQ(9, IntFromString("-0x1fg", 0, &slen), Big.Int(-(1*16 + 15)));
	TEST(10, slen == 5);
}


func RatConv() {
	test_msg = "RatConv";
	var slen int;
	RAT_EQ(0, RatFromString("0", 0, nil), rat_zero);
	RAT_EQ(1, RatFromString("0/1", 0, nil), rat_zero);
	RAT_EQ(2, RatFromString("0/01", 0, nil), rat_zero);
	RAT_EQ(3, RatFromString("0x14/10", 0, &slen), rat_two);
	TEST(4, slen == 7);
	RAT_EQ(5, RatFromString("0.", 0, nil), rat_zero);
	RAT_EQ(6, RatFromString("0.001f", 10, nil), Big.Rat(1, 1000));
	RAT_EQ(7, RatFromString("10101.0101", 2, nil), Big.Rat(0x155, 1<<4));
	RAT_EQ(8, RatFromString("-0003.145926", 10, &slen), Big.Rat(-3145926, 1000000));
	TEST(9, slen == 12);
}


func Add(x, y *Big.Natural) *Big.Natural {
	z1 := x.Add(y);
	z2 := y.Add(x);
	if z1.Cmp(z2) != 0 {
		println("addition not symmetric");
		println("x =", x.String(10));
		println("y =", y.String(10));
		panic();
	}
	return z1;
}


func Sum(n uint, scale *Big.Natural) *Big.Natural {
	s := nat_zero;
	for ; n > 0; n-- {
		s = Add(s, Big.Nat(n).Mul(scale));
	}
	return s;
}


func NatAdd() {
	test_msg = "NatAddA";
	NAT_EQ(0, Add(nat_zero, nat_zero), nat_zero);
	NAT_EQ(1, Add(nat_zero, c), c);

	test_msg = "NatAddB";
	for i := uint(0); i < 100; i++ {
		t := Big.Nat(i);
		NAT_EQ(i, Sum(i, c), t.Mul(t).Add(t).Shr(1).Mul(c));
	}
}


func Mul(x, y *Big.Natural) *Big.Natural {
	z1 := x.Mul(y);
	z2 := y.Mul(x);
	if z1.Cmp(z2) != 0 {
		println("multiplication not symmetric");
		println("x =", x.String(10));
		println("y =", y.String(10));
		panic();
	}
	if !x.IsZero() && z1.Div(x).Cmp(y) != 0 {
		println("multiplication/division not inverse (A)");
		println("x =", x.String(10));
		println("y =", y.String(10));
		panic();
	}
	if !y.IsZero() && z1.Div(y).Cmp(x) != 0 {
		println("multiplication/division not inverse (B)");
		println("x =", x.String(10));
		println("y =", y.String(10));
		panic();
	}
	return z1;
}


func NatSub() {
	test_msg = "NatSubA";
	NAT_EQ(0, nat_zero.Sub(nat_zero), nat_zero);
	NAT_EQ(1, c.Sub(nat_zero), c);

	test_msg = "NatSubB";
	for i := uint(0); i < 100; i++ {
		t := Sum(i, c);
		for j := uint(0); j <= i; j++ {
			t = t.Sub(Mul(Big.Nat(j), c));
		}
		NAT_EQ(i, t, nat_zero);
	}
}


func NatMul() {
	test_msg = "NatMulA";
	NAT_EQ(0, Mul(c, nat_zero), nat_zero);
	NAT_EQ(1, Mul(c, nat_one), c);

	test_msg = "NatMulB";
	NAT_EQ(0, b.Mul(Big.MulRange(0, 100)), nat_zero);
	NAT_EQ(1, b.Mul(Big.MulRange(21, 100)), c);

	test_msg = "NatMulC";
	const n = 100;
	p := b.Mul(c).Shl(n);
	for i := uint(0); i < n; i++ {
		NAT_EQ(i, Mul(b.Shl(i), c.Shl(n-i)), p);
	}
}


func NatDiv() {
	test_msg = "NatDivA";
	NAT_EQ(0, c.Div(nat_one), c);
	NAT_EQ(1, c.Div(Big.Nat(100)), Big.Fact(99));
	NAT_EQ(2, b.Div(c), nat_zero);
	NAT_EQ(4, nat_one.Shl(100).Div(nat_one.Shl(90)), nat_one.Shl(10));
	NAT_EQ(5, c.Div(b), Big.MulRange(21, 100));

	test_msg = "NatDivB";
	const n = 100;
	p := Big.Fact(n);
	for i := uint(0); i < n; i++ {
		NAT_EQ(i, p.Div(Big.MulRange(1, i)), Big.MulRange(i+1, n));
	}
}


func IntQuoRem() {
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
		x, y := Big.Int(e.x).Mul(ip), Big.Int(e.y).Mul(ip);
		q, r := Big.Int(e.q), Big.Int(e.r).Mul(ip);
		qq, rr := x.QuoRem(y);
		INT_EQ(4*i+0, x.Quo(y), q);
		INT_EQ(4*i+1, x.Rem(y), r);
		INT_EQ(4*i+2, qq, q);
		INT_EQ(4*i+3, rr, r);
	}
}


func IntDivMod() {
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
		x, y := Big.Int(e.x).Mul(ip), Big.Int(e.y).Mul(ip);
		q, r := Big.Int(e.q), Big.Int(e.r).Mul(ip);
		qq, rr := x.DivMod(y);
		INT_EQ(4*i+0, x.Div(y), q);
		INT_EQ(4*i+1, x.Mod(y), r);
		INT_EQ(4*i+2, qq, q);
		INT_EQ(4*i+3, rr, r);
	}
}


func NatMod() {
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


func NatShift() {
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
		f := Big.Nat(1<<m);
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


func IntShift() {
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
		f := Big.Int(1<<m);
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
	//INT_EQ(0, Big.Int(-43).Shr(1), Big.Int(-43 >> 1));
	//INT_EQ(1, ip.Neg().Shr(10), ip.Neg().Div(Big.Int(1).Shl(10)));
}


func NatCmp() {
	test_msg = "NatCmp";
	TEST(0, a.Cmp(a) == 0);
	TEST(1, a.Cmp(b) < 0);
	TEST(2, b.Cmp(a) > 0);
	TEST(3, a.Cmp(c) < 0);
	d := c.Add(b);
	TEST(4, c.Cmp(d) < 0);
	TEST(5, d.Cmp(c) > 0);
}


func NatLog2() {
	test_msg = "NatLog2A";
	TEST(0, nat_one.Log2() == 0);
	TEST(1, nat_two.Log2() == 1);
	TEST(2, Big.Nat(3).Log2() == 1);
	TEST(3, Big.Nat(4).Log2() == 2);
	
	test_msg = "NatLog2B";
	for i := uint(0); i < 100; i++ {
		TEST(i, nat_one.Shl(i).Log2() == i);
	}
}


func NatGcd() {
	test_msg = "NatGcdA";
	f := Big.Nat(99991);
	NAT_EQ(0, b.Mul(f).Gcd(c.Mul(f)), Big.MulRange(1, 20).Mul(f));
}


func NatPow() {
	test_msg = "NatPowA";
	NAT_EQ(0, nat_two.Pow(0), nat_one);

	test_msg = "NatPowB";
	for i := uint(0); i < 100; i++ {
		NAT_EQ(i, nat_two.Pow(i), nat_one.Shl(i));
	}
}


func NatPop() {
	test_msg = "NatPopA";
	TEST(0, nat_zero.Pop() == 0);
	TEST(1, nat_one.Pop() == 1);
	TEST(2, Big.Nat(10).Pop() == 2);
	TEST(3, Big.Nat(30).Pop() == 4);
	TEST(4, Big.Nat(0x1248f).Shl(33).Pop() == 8);

	test_msg = "NatPopB";
	for i := uint(0); i < 100; i++ {
		TEST(i, nat_one.Shl(i).Sub(nat_one).Pop() == i);
	}
}


func main() {
	// Naturals
	NatConv();
	NatAdd();
	NatSub();
	NatMul();
	NatDiv();
	NatMod();
	NatShift();
	NatCmp();
	NatLog2();
	NatGcd();
	NatPow();
	NatPop();

	// Integers
	// TODO add more tests
	IntConv();
	IntQuoRem();
	IntDivMod();
	IntShift();

	// Rationals
	// TODO add more tests
	RatConv();
}

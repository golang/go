// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bignum

import (
	"fmt";
	"testing";
)

const (
	sa	= "991";
	sb	= "2432902008176640000";	// 20!
	sc	= "933262154439441526816992388562667004907159682643816214685929"
		"638952175999932299156089414639761565182862536979208272237582"
		"51185210916864000000000000000000000000";	// 100!
	sp	= "170141183460469231731687303715884105727";	// prime
)

func natFromString(s string, base uint, slen *int) Natural {
	x, _, len := NatFromString(s, base);
	if slen != nil {
		*slen = len
	}
	return x;
}


func intFromString(s string, base uint, slen *int) *Integer {
	x, _, len := IntFromString(s, base);
	if slen != nil {
		*slen = len
	}
	return x;
}


func ratFromString(s string, base uint, slen *int) *Rational {
	x, _, len := RatFromString(s, base);
	if slen != nil {
		*slen = len
	}
	return x;
}


var (
	nat_zero	= Nat(0);
	nat_one		= Nat(1);
	nat_two		= Nat(2);
	a		= natFromString(sa, 10, nil);
	b		= natFromString(sb, 10, nil);
	c		= natFromString(sc, 10, nil);
	p		= natFromString(sp, 10, nil);
	int_zero	= Int(0);
	int_one		= Int(1);
	int_two		= Int(2);
	ip		= intFromString(sp, 10, nil);
	rat_zero	= Rat(0, 1);
	rat_half	= Rat(1, 2);
	rat_one		= Rat(1, 1);
	rat_two		= Rat(2, 1);
)


var test_msg string
var tester *testing.T

func test(n uint, b bool) {
	if !b {
		tester.Fatalf("TEST failed: %s (%d)", test_msg, n)
	}
}


func nat_eq(n uint, x, y Natural) {
	if x.Cmp(y) != 0 {
		tester.Fatalf("TEST failed: %s (%d)\nx = %v\ny = %v", test_msg, n, &x, &y)
	}
}


func int_eq(n uint, x, y *Integer) {
	if x.Cmp(y) != 0 {
		tester.Fatalf("TEST failed: %s (%d)\nx = %v\ny = %v", test_msg, n, x, y)
	}
}


func rat_eq(n uint, x, y *Rational) {
	if x.Cmp(y) != 0 {
		tester.Fatalf("TEST failed: %s (%d)\nx = %v\ny = %v", test_msg, n, x, y)
	}
}


func TestNatConv(t *testing.T) {
	tester = t;
	test_msg = "NatConvA";
	type entry1 struct {
		x	uint64;
		s	string;
	}
	tab := []entry1{
		entry1{0, "0"},
		entry1{255, "255"},
		entry1{65535, "65535"},
		entry1{4294967295, "4294967295"},
		entry1{18446744073709551615, "18446744073709551615"},
	};
	for i, e := range tab {
		test(100+uint(i), Nat(e.x).String() == e.s);
		test(200+uint(i), natFromString(e.s, 0, nil).Value() == e.x);
	}

	test_msg = "NatConvB";
	for i := uint(0); i < 100; i++ {
		test(i, Nat(uint64(i)).String() == fmt.Sprintf("%d", i))
	}

	test_msg = "NatConvC";
	z := uint64(7);
	for i := uint(0); i <= 64; i++ {
		test(i, Nat(z).Value() == z);
		z <<= 1;
	}

	test_msg = "NatConvD";
	nat_eq(0, a, Nat(991));
	nat_eq(1, b, Fact(20));
	nat_eq(2, c, Fact(100));
	test(3, a.String() == sa);
	test(4, b.String() == sb);
	test(5, c.String() == sc);

	test_msg = "NatConvE";
	var slen int;
	nat_eq(10, natFromString("0", 0, nil), nat_zero);
	nat_eq(11, natFromString("123", 0, nil), Nat(123));
	nat_eq(12, natFromString("077", 0, nil), Nat(7*8+7));
	nat_eq(13, natFromString("0x1f", 0, nil), Nat(1*16+15));
	nat_eq(14, natFromString("0x1fg", 0, &slen), Nat(1*16+15));
	test(4, slen == 4);

	test_msg = "NatConvF";
	tmp := c.Mul(c);
	for base := uint(2); base <= 16; base++ {
		nat_eq(base, natFromString(tmp.ToString(base), base, nil), tmp)
	}

	test_msg = "NatConvG";
	x := Nat(100);
	y, _, _ := NatFromString(fmt.Sprintf("%b", &x), 2);
	nat_eq(100, y, x);
}


func abs(x int64) uint64 {
	if x < 0 {
		x = -x
	}
	return uint64(x);
}


func TestIntConv(t *testing.T) {
	tester = t;
	test_msg = "IntConvA";
	type entry2 struct {
		x	int64;
		s	string;
	}
	tab := []entry2{
		entry2{0, "0"},
		entry2{-128, "-128"},
		entry2{127, "127"},
		entry2{-32768, "-32768"},
		entry2{32767, "32767"},
		entry2{-2147483648, "-2147483648"},
		entry2{2147483647, "2147483647"},
		entry2{-9223372036854775808, "-9223372036854775808"},
		entry2{9223372036854775807, "9223372036854775807"},
	};
	for i, e := range tab {
		test(100+uint(i), Int(e.x).String() == e.s);
		test(200+uint(i), intFromString(e.s, 0, nil).Value() == e.x);
		test(300+uint(i), Int(e.x).Abs().Value() == abs(e.x));
	}

	test_msg = "IntConvB";
	var slen int;
	int_eq(0, intFromString("0", 0, nil), int_zero);
	int_eq(1, intFromString("-0", 0, nil), int_zero);
	int_eq(2, intFromString("123", 0, nil), Int(123));
	int_eq(3, intFromString("-123", 0, nil), Int(-123));
	int_eq(4, intFromString("077", 0, nil), Int(7*8+7));
	int_eq(5, intFromString("-077", 0, nil), Int(-(7*8 + 7)));
	int_eq(6, intFromString("0x1f", 0, nil), Int(1*16+15));
	int_eq(7, intFromString("-0x1f", 0, &slen), Int(-(1*16 + 15)));
	test(7, slen == 5);
	int_eq(8, intFromString("+0x1f", 0, &slen), Int(+(1*16 + 15)));
	test(8, slen == 5);
	int_eq(9, intFromString("0x1fg", 0, &slen), Int(1*16+15));
	test(9, slen == 4);
	int_eq(10, intFromString("-0x1fg", 0, &slen), Int(-(1*16 + 15)));
	test(10, slen == 5);
}


func TestRatConv(t *testing.T) {
	tester = t;
	test_msg = "RatConv";
	var slen int;
	rat_eq(0, ratFromString("0", 0, nil), rat_zero);
	rat_eq(1, ratFromString("0/1", 0, nil), rat_zero);
	rat_eq(2, ratFromString("0/01", 0, nil), rat_zero);
	rat_eq(3, ratFromString("0x14/10", 0, &slen), rat_two);
	test(4, slen == 7);
	rat_eq(5, ratFromString("0.", 0, nil), rat_zero);
	rat_eq(6, ratFromString("0.001f", 10, nil), Rat(1, 1000));
	rat_eq(7, ratFromString(".1", 0, nil), Rat(1, 10));
	rat_eq(8, ratFromString("10101.0101", 2, nil), Rat(0x155, 1<<4));
	rat_eq(9, ratFromString("-0003.145926", 10, &slen), Rat(-3145926, 1000000));
	test(10, slen == 12);
	rat_eq(11, ratFromString("1e2", 0, nil), Rat(100, 1));
	rat_eq(12, ratFromString("1e-2", 0, nil), Rat(1, 100));
	rat_eq(13, ratFromString("1.1e2", 0, nil), Rat(110, 1));
	rat_eq(14, ratFromString(".1e2x", 0, &slen), Rat(10, 1));
	test(15, slen == 4);
}


func add(x, y Natural) Natural {
	z1 := x.Add(y);
	z2 := y.Add(x);
	if z1.Cmp(z2) != 0 {
		tester.Fatalf("addition not symmetric:\n\tx = %v\n\ty = %t", x, y)
	}
	return z1;
}


func sum(n uint64, scale Natural) Natural {
	s := nat_zero;
	for ; n > 0; n-- {
		s = add(s, Nat(n).Mul(scale))
	}
	return s;
}


func TestNatAdd(t *testing.T) {
	tester = t;
	test_msg = "NatAddA";
	nat_eq(0, add(nat_zero, nat_zero), nat_zero);
	nat_eq(1, add(nat_zero, c), c);

	test_msg = "NatAddB";
	for i := uint64(0); i < 100; i++ {
		t := Nat(i);
		nat_eq(uint(i), sum(i, c), t.Mul(t).Add(t).Shr(1).Mul(c));
	}
}


func mul(x, y Natural) Natural {
	z1 := x.Mul(y);
	z2 := y.Mul(x);
	if z1.Cmp(z2) != 0 {
		tester.Fatalf("multiplication not symmetric:\n\tx = %v\n\ty = %t", x, y)
	}
	if !x.IsZero() && z1.Div(x).Cmp(y) != 0 {
		tester.Fatalf("multiplication/division not inverse (A):\n\tx = %v\n\ty = %t", x, y)
	}
	if !y.IsZero() && z1.Div(y).Cmp(x) != 0 {
		tester.Fatalf("multiplication/division not inverse (B):\n\tx = %v\n\ty = %t", x, y)
	}
	return z1;
}


func TestNatSub(t *testing.T) {
	tester = t;
	test_msg = "NatSubA";
	nat_eq(0, nat_zero.Sub(nat_zero), nat_zero);
	nat_eq(1, c.Sub(nat_zero), c);

	test_msg = "NatSubB";
	for i := uint64(0); i < 100; i++ {
		t := sum(i, c);
		for j := uint64(0); j <= i; j++ {
			t = t.Sub(mul(Nat(j), c))
		}
		nat_eq(uint(i), t, nat_zero);
	}
}


func TestNatMul(t *testing.T) {
	tester = t;
	test_msg = "NatMulA";
	nat_eq(0, mul(c, nat_zero), nat_zero);
	nat_eq(1, mul(c, nat_one), c);

	test_msg = "NatMulB";
	nat_eq(0, b.Mul(MulRange(0, 100)), nat_zero);
	nat_eq(1, b.Mul(MulRange(21, 100)), c);

	test_msg = "NatMulC";
	const n = 100;
	p := b.Mul(c).Shl(n);
	for i := uint(0); i < n; i++ {
		nat_eq(i, mul(b.Shl(i), c.Shl(n-i)), p)
	}
}


func TestNatDiv(t *testing.T) {
	tester = t;
	test_msg = "NatDivA";
	nat_eq(0, c.Div(nat_one), c);
	nat_eq(1, c.Div(Nat(100)), Fact(99));
	nat_eq(2, b.Div(c), nat_zero);
	nat_eq(4, nat_one.Shl(100).Div(nat_one.Shl(90)), nat_one.Shl(10));
	nat_eq(5, c.Div(b), MulRange(21, 100));

	test_msg = "NatDivB";
	const n = 100;
	p := Fact(n);
	for i := uint(0); i < n; i++ {
		nat_eq(100+i, p.Div(MulRange(1, i)), MulRange(i+1, n))
	}
}


func TestIntQuoRem(t *testing.T) {
	tester = t;
	test_msg = "IntQuoRem";
	type T struct {
		x, y, q, r int64;
	}
	a := []T{
		T{+8, +3, +2, +2},
		T{+8, -3, -2, +2},
		T{-8, +3, -2, -2},
		T{-8, -3, +2, -2},
		T{+1, +2, 0, +1},
		T{+1, -2, 0, +1},
		T{-1, +2, 0, -1},
		T{-1, -2, 0, -1},
	};
	for i := uint(0); i < uint(len(a)); i++ {
		e := &a[i];
		x, y := Int(e.x).Mul(ip), Int(e.y).Mul(ip);
		q, r := Int(e.q), Int(e.r).Mul(ip);
		qq, rr := x.QuoRem(y);
		int_eq(4*i+0, x.Quo(y), q);
		int_eq(4*i+1, x.Rem(y), r);
		int_eq(4*i+2, qq, q);
		int_eq(4*i+3, rr, r);
	}
}


func TestIntDivMod(t *testing.T) {
	tester = t;
	test_msg = "IntDivMod";
	type T struct {
		x, y, q, r int64;
	}
	a := []T{
		T{+8, +3, +2, +2},
		T{+8, -3, -2, +2},
		T{-8, +3, -3, +1},
		T{-8, -3, +3, +1},
		T{+1, +2, 0, +1},
		T{+1, -2, 0, +1},
		T{-1, +2, -1, +1},
		T{-1, -2, +1, +1},
	};
	for i := uint(0); i < uint(len(a)); i++ {
		e := &a[i];
		x, y := Int(e.x).Mul(ip), Int(e.y).Mul(ip);
		q, r := Int(e.q), Int(e.r).Mul(ip);
		qq, rr := x.DivMod(y);
		int_eq(4*i+0, x.Div(y), q);
		int_eq(4*i+1, x.Mod(y), r);
		int_eq(4*i+2, qq, q);
		int_eq(4*i+3, rr, r);
	}
}


func TestNatMod(t *testing.T) {
	tester = t;
	test_msg = "NatModA";
	for i := uint(0); ; i++ {
		d := nat_one.Shl(i);
		if d.Cmp(c) < 0 {
			nat_eq(i, c.Add(d).Mod(c), d)
		} else {
			nat_eq(i, c.Add(d).Div(c), nat_two);
			nat_eq(i, c.Add(d).Mod(c), d.Sub(c));
			break;
		}
	}
}


func TestNatShift(t *testing.T) {
	tester = t;
	test_msg = "NatShift1L";
	test(0, b.Shl(0).Cmp(b) == 0);
	test(1, c.Shl(1).Cmp(c) > 0);

	test_msg = "NatShift1R";
	test(3, b.Shr(0).Cmp(b) == 0);
	test(4, c.Shr(1).Cmp(c) < 0);

	test_msg = "NatShift2";
	for i := uint(0); i < 100; i++ {
		test(i, c.Shl(i).Shr(i).Cmp(c) == 0)
	}

	test_msg = "NatShift3L";
	{
		const m = 3;
		p := b;
		f := Nat(1 << m);
		for i := uint(0); i < 100; i++ {
			nat_eq(i, b.Shl(i*m), p);
			p = mul(p, f);
		}
	}

	test_msg = "NatShift3R";
	{
		p := c;
		for i := uint(0); !p.IsZero(); i++ {
			nat_eq(i, c.Shr(i), p);
			p = p.Shr(1);
		}
	}
}


func TestIntShift(t *testing.T) {
	tester = t;
	test_msg = "IntShift1L";
	test(0, ip.Shl(0).Cmp(ip) == 0);
	test(1, ip.Shl(1).Cmp(ip) > 0);

	test_msg = "IntShift1R";
	test(0, ip.Shr(0).Cmp(ip) == 0);
	test(1, ip.Shr(1).Cmp(ip) < 0);

	test_msg = "IntShift2";
	for i := uint(0); i < 100; i++ {
		test(i, ip.Shl(i).Shr(i).Cmp(ip) == 0)
	}

	test_msg = "IntShift3L";
	{
		const m = 3;
		p := ip;
		f := Int(1 << m);
		for i := uint(0); i < 100; i++ {
			int_eq(i, ip.Shl(i*m), p);
			p = p.Mul(f);
		}
	}

	test_msg = "IntShift3R";
	{
		p := ip;
		for i := uint(0); p.IsPos(); i++ {
			int_eq(i, ip.Shr(i), p);
			p = p.Shr(1);
		}
	}

	test_msg = "IntShift4R";
	int_eq(0, Int(-43).Shr(1), Int(-43>>1));
	int_eq(0, Int(-1024).Shr(100), Int(-1));
	int_eq(1, ip.Neg().Shr(10), ip.Neg().Div(Int(1).Shl(10)));
}


func TestNatBitOps(t *testing.T) {
	tester = t;

	x := uint64(0xf08e6f56bd8c3941);
	y := uint64(0x3984ef67834bc);

	bx := Nat(x);
	by := Nat(y);

	test_msg = "NatAnd";
	bz := Nat(x & y);
	for i := uint(0); i < 100; i++ {
		nat_eq(i, bx.Shl(i).And(by.Shl(i)), bz.Shl(i))
	}

	test_msg = "NatAndNot";
	bz = Nat(x &^ y);
	for i := uint(0); i < 100; i++ {
		nat_eq(i, bx.Shl(i).AndNot(by.Shl(i)), bz.Shl(i))
	}

	test_msg = "NatOr";
	bz = Nat(x | y);
	for i := uint(0); i < 100; i++ {
		nat_eq(i, bx.Shl(i).Or(by.Shl(i)), bz.Shl(i))
	}

	test_msg = "NatXor";
	bz = Nat(x ^ y);
	for i := uint(0); i < 100; i++ {
		nat_eq(i, bx.Shl(i).Xor(by.Shl(i)), bz.Shl(i))
	}
}


func TestIntBitOps1(t *testing.T) {
	tester = t;
	test_msg = "IntBitOps1";
	type T struct {
		x, y int64;
	}
	a := []T{
		T{+7, +3},
		T{+7, -3},
		T{-7, +3},
		T{-7, -3},
	};
	for i := uint(0); i < uint(len(a)); i++ {
		e := &a[i];
		int_eq(4*i+0, Int(e.x).And(Int(e.y)), Int(e.x&e.y));
		int_eq(4*i+1, Int(e.x).AndNot(Int(e.y)), Int(e.x&^e.y));
		int_eq(4*i+2, Int(e.x).Or(Int(e.y)), Int(e.x|e.y));
		int_eq(4*i+3, Int(e.x).Xor(Int(e.y)), Int(e.x^e.y));
	}
}


func TestIntBitOps2(t *testing.T) {
	tester = t;

	test_msg = "IntNot";
	int_eq(0, Int(-2).Not(), Int(1));
	int_eq(0, Int(-1).Not(), Int(0));
	int_eq(0, Int(0).Not(), Int(-1));
	int_eq(0, Int(1).Not(), Int(-2));
	int_eq(0, Int(2).Not(), Int(-3));

	test_msg = "IntAnd";
	for x := int64(-15); x < 5; x++ {
		bx := Int(x);
		for y := int64(-5); y < 15; y++ {
			by := Int(y);
			for i := uint(50); i < 70; i++ {	// shift across 64bit boundary
				int_eq(i, bx.Shl(i).And(by.Shl(i)), Int(x&y).Shl(i))
			}
		}
	}

	test_msg = "IntAndNot";
	for x := int64(-15); x < 5; x++ {
		bx := Int(x);
		for y := int64(-5); y < 15; y++ {
			by := Int(y);
			for i := uint(50); i < 70; i++ {	// shift across 64bit boundary
				int_eq(2*i+0, bx.Shl(i).AndNot(by.Shl(i)), Int(x&^y).Shl(i));
				int_eq(2*i+1, bx.Shl(i).And(by.Shl(i).Not()), Int(x&^y).Shl(i));
			}
		}
	}

	test_msg = "IntOr";
	for x := int64(-15); x < 5; x++ {
		bx := Int(x);
		for y := int64(-5); y < 15; y++ {
			by := Int(y);
			for i := uint(50); i < 70; i++ {	// shift across 64bit boundary
				int_eq(i, bx.Shl(i).Or(by.Shl(i)), Int(x|y).Shl(i))
			}
		}
	}

	test_msg = "IntXor";
	for x := int64(-15); x < 5; x++ {
		bx := Int(x);
		for y := int64(-5); y < 15; y++ {
			by := Int(y);
			for i := uint(50); i < 70; i++ {	// shift across 64bit boundary
				int_eq(i, bx.Shl(i).Xor(by.Shl(i)), Int(x^y).Shl(i))
			}
		}
	}
}


func TestNatCmp(t *testing.T) {
	tester = t;
	test_msg = "NatCmp";
	test(0, a.Cmp(a) == 0);
	test(1, a.Cmp(b) < 0);
	test(2, b.Cmp(a) > 0);
	test(3, a.Cmp(c) < 0);
	d := c.Add(b);
	test(4, c.Cmp(d) < 0);
	test(5, d.Cmp(c) > 0);
}


func TestNatLog2(t *testing.T) {
	tester = t;
	test_msg = "NatLog2A";
	test(0, nat_one.Log2() == 0);
	test(1, nat_two.Log2() == 1);
	test(2, Nat(3).Log2() == 1);
	test(3, Nat(4).Log2() == 2);

	test_msg = "NatLog2B";
	for i := uint(0); i < 100; i++ {
		test(i, nat_one.Shl(i).Log2() == i)
	}
}


func TestNatGcd(t *testing.T) {
	tester = t;
	test_msg = "NatGcdA";
	f := Nat(99991);
	nat_eq(0, b.Mul(f).Gcd(c.Mul(f)), MulRange(1, 20).Mul(f));
}


func TestNatPow(t *testing.T) {
	tester = t;
	test_msg = "NatPowA";
	nat_eq(0, nat_two.Pow(0), nat_one);

	test_msg = "NatPowB";
	for i := uint(0); i < 100; i++ {
		nat_eq(i, nat_two.Pow(i), nat_one.Shl(i))
	}
}


func TestNatPop(t *testing.T) {
	tester = t;
	test_msg = "NatPopA";
	test(0, nat_zero.Pop() == 0);
	test(1, nat_one.Pop() == 1);
	test(2, Nat(10).Pop() == 2);
	test(3, Nat(30).Pop() == 4);
	test(4, Nat(0x1248f).Shl(33).Pop() == 8);

	test_msg = "NatPopB";
	for i := uint(0); i < 100; i++ {
		test(i, nat_one.Shl(i).Sub(nat_one).Pop() == i)
	}
}

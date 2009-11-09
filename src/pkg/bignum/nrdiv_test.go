// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements Newton-Raphson division and uses
// it as an additional test case for bignum.
//
// Division of x/y is achieved by computing r = 1/y to
// obtain the quotient q = x*r = x*(1/y) = x/y. The
// reciprocal r is the solution for f(x) = 1/x - y and
// the solution is approximated through iteration. The
// iteration does not require division.

package bignum

import "testing"


// An fpNat is a Natural scaled by a power of two
// (an unsigned floating point representation). The
// value of an fpNat x is x.m * 2^x.e .
//
type fpNat struct {
	m	Natural;
	e	int;
}


// sub computes x - y.
func (x fpNat) sub(y fpNat) fpNat {
	switch d := x.e - y.e; {
	case d < 0:
		return fpNat{x.m.Sub(y.m.Shl(uint(-d))), x.e}
	case d > 0:
		return fpNat{x.m.Shl(uint(d)).Sub(y.m), y.e}
	}
	return fpNat{x.m.Sub(y.m), x.e};
}


// mul2 computes x*2.
func (x fpNat) mul2() fpNat	{ return fpNat{x.m, x.e + 1} }


// mul computes x*y.
func (x fpNat) mul(y fpNat) fpNat	{ return fpNat{x.m.Mul(y.m), x.e + y.e} }


// mant computes the (possibly truncated) Natural representation
// of an fpNat x.
//
func (x fpNat) mant() Natural {
	switch {
	case x.e > 0:
		return x.m.Shl(uint(x.e))
	case x.e < 0:
		return x.m.Shr(uint(-x.e))
	}
	return x.m;
}


// nrDivEst computes an estimate of the quotient q = x0/y0 and returns q.
// q may be too small (usually by 1).
//
func nrDivEst(x0, y0 Natural) Natural {
	if y0.IsZero() {
		panic("division by zero");
		return nil;
	}
	// y0 > 0

	if y0.Cmp(Nat(1)) == 0 {
		return x0
	}
	// y0 > 1

	switch d := x0.Cmp(y0); {
	case d < 0:
		return Nat(0)
	case d == 0:
		return Nat(1)
	}
	// x0 > y0 > 1

	// Determine maximum result length.
	maxLen := int(x0.Log2() - y0.Log2() + 1);

	// In the following, each number x is represented
	// as a mantissa x.m and an exponent x.e such that
	// x = xm * 2^x.e.
	x := fpNat{x0, 0};
	y := fpNat{y0, 0};

	// Determine a scale factor f = 2^e such that
	// 0.5 <= y/f == y*(2^-e) < 1.0
	// and scale y accordingly.
	e := int(y.m.Log2())+1;
	y.e -= e;

	// t1
	var c = 2.9142;
	const n = 14;
	t1 := fpNat{Nat(uint64(c*(1<<n))), -n};

	// Compute initial value r0 for the reciprocal of y/f.
	// r0 = t1 - 2*y
	r := t1.sub(y.mul2());
	two := fpNat{Nat(2), 0};

	// Newton-Raphson iteration
	p := Nat(0);
	for i := 0; ; i++ {
		// check if we are done
		// TODO: Need to come up with a better test here
		//       as it will reduce computation time significantly.
		// q = x*r/f
		q := x.mul(r);
		q.e -= e;
		res := q.mant();
		if res.Cmp(p) == 0 {
			return res
		}
		p = res;

		// r' = r*(2 - y*r)
		r = r.mul(two.sub(y.mul(r)));

		// reduce mantissa size
		// TODO: Find smaller bound as it will reduce
		//       computation time massively.
		d := int(r.m.Log2() + 1)-maxLen;
		if d > 0 {
			r = fpNat{r.m.Shr(uint(d)), r.e + d}
		}
	}

	panic("unreachable");
	return nil;
}


func nrdiv(x, y Natural) (q, r Natural) {
	q = nrDivEst(x, y);
	r = x.Sub(y.Mul(q));
	// if r is too large, correct q and r
	// (usually one iteration)
	for r.Cmp(y) >= 0 {
		q = q.Add(Nat(1));
		r = r.Sub(y);
	}
	return;
}


func div(t *testing.T, x, y Natural) {
	q, r := nrdiv(x, y);
	qx, rx := x.DivMod(y);
	if q.Cmp(qx) != 0 {
		t.Errorf("x = %s, y = %s, got q = %s, want q = %s", x, y, q, qx)
	}
	if r.Cmp(rx) != 0 {
		t.Errorf("x = %s, y = %s, got r = %s, want r = %s", x, y, r, rx)
	}
}


func idiv(t *testing.T, x0, y0 uint64)	{ div(t, Nat(x0), Nat(y0)) }


func TestNRDiv(t *testing.T) {
	idiv(t, 17, 18);
	idiv(t, 17, 17);
	idiv(t, 17, 1);
	idiv(t, 17, 16);
	idiv(t, 17, 10);
	idiv(t, 17, 9);
	idiv(t, 17, 8);
	idiv(t, 17, 5);
	idiv(t, 17, 3);
	idiv(t, 1025, 512);
	idiv(t, 7489595, 2);
	idiv(t, 5404679459, 78495);
	idiv(t, 7484890589595, 7484890589594);
	div(t, Fact(100), Fact(91));
	div(t, Fact(1000), Fact(991));
	//div(t, Fact(10000), Fact(9991));  // takes too long - disabled for now
}

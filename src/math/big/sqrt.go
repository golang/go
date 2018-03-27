// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import "math"

var (
	half  = NewFloat(0.5)
	two   = NewFloat(2.0)
	three = NewFloat(3.0)
)

// Sqrt sets z to the rounded square root of x, and returns it.
//
// If z's precision is 0, it is changed to x's precision before the
// operation. Rounding is performed according to z's precision and
// rounding mode.
//
// The function panics if z < 0. The value of z is undefined in that
// case.
func (z *Float) Sqrt(x *Float) *Float {
	if debugFloat {
		x.validate()
	}

	if z.prec == 0 {
		z.prec = x.prec
	}

	if x.Sign() == -1 {
		// following IEEE754-2008 (section 7.2)
		panic(ErrNaN{"square root of negative operand"})
	}

	// handle ±0 and +∞
	if x.form != finite {
		z.acc = Exact
		z.form = x.form
		z.neg = x.neg // IEEE754-2008 requires √±0 = ±0
		return z
	}

	// MantExp sets the argument's precision to the receiver's, and
	// when z.prec > x.prec this will lower z.prec. Restore it after
	// the MantExp call.
	prec := z.prec
	b := x.MantExp(z)
	z.prec = prec

	// Compute √(z·2**b) as
	//   √( z)·2**(½b)     if b is even
	//   √(2z)·2**(⌊½b⌋)   if b > 0 is odd
	//   √(½z)·2**(⌈½b⌉)   if b < 0 is odd
	switch b % 2 {
	case 0:
		// nothing to do
	case 1:
		z.Mul(two, z)
	case -1:
		z.Mul(half, z)
	}
	// 0.25 <= z < 2.0

	// Solving x² - z = 0 directly requires a Quo call, but it's
	// faster for small precisions.
	//
	// Solving 1/x² - z = 0 avoids the Quo call and is much faster for
	// high precisions.
	//
	// 128bit precision is an empirically chosen threshold.
	if z.prec <= 128 {
		z.sqrtDirect(z)
	} else {
		z.sqrtInverse(z)
	}

	// re-attach halved exponent
	return z.SetMantExp(z, b/2)
}

// Compute √x (up to prec 128) by solving
//   t² - x = 0
// for t, starting with a 53 bits precision guess from math.Sqrt and
// then using at most two iterations of Newton's method.
func (z *Float) sqrtDirect(x *Float) {
	// let
	//   f(t) = t² - x
	// then
	//   g(t) = f(t)/f'(t) = ½(t² - x)/t
	// and the next guess is given by
	//   t2 = t - g(t) = ½(t² + x)/t
	u := new(Float)
	ng := func(t *Float) *Float {
		u.prec = t.prec
		u.Mul(t, t)        // u = t²
		u.Add(u, x)        //   = t² + x
		u.Mul(half, u)     //   = ½(t² + x)
		return t.Quo(u, t) //   = ½(t² + x)/t
	}

	xf, _ := x.Float64()
	sq := NewFloat(math.Sqrt(xf))

	switch {
	case z.prec > 128:
		panic("sqrtDirect: only for z.prec <= 128")
	case z.prec > 64:
		sq.prec *= 2
		sq = ng(sq)
		fallthrough
	default:
		sq.prec *= 2
		sq = ng(sq)
	}

	z.Set(sq)
}

// Compute √x (to z.prec precision) by solving
//   1/t² - x = 0
// for t (using Newton's method), and then inverting.
func (z *Float) sqrtInverse(x *Float) {
	// let
	//   f(t) = 1/t² - x
	// then
	//   g(t) = f(t)/f'(t) = -½t(1 - xt²)
	// and the next guess is given by
	//   t2 = t - g(t) = ½t(3 - xt²)
	u := newFloat(z.prec)
	v := newFloat(z.prec)
	ng := func(t *Float) *Float {
		u.prec = t.prec
		v.prec = t.prec
		u.Mul(t, t)           // u = t²
		u.Mul(x, u)           //   = xt²
		v.Sub(three, u)       // v = 3 - xt²
		u.Mul(t, v)           // u = t(3 - xt²)
		return t.Mul(half, u) //   = ½t(3 - xt²)
	}

	xf, _ := x.Float64()
	sqi := newFloat(z.prec)
	sqi.SetFloat64(1 / math.Sqrt(xf))
	for prec := z.prec + 32; sqi.prec < prec; {
		sqi.prec *= 2
		sqi = ng(sqi)
	}
	// sqi = 1/√x

	// x/√x = √x
	z.Mul(x, sqi)
}

// newFloat returns a new *Float with space for twice the given
// precision.
func newFloat(prec2 uint32) *Float {
	z := new(Float)
	// nat.make ensures the slice length is > 0
	z.mant = z.mant.make(int(prec2/_W) * 2)
	return z
}

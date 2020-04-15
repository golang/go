// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import (
	"math"
	"sync"
)

var threeOnce struct {
	sync.Once
	v *Float
}

func three() *Float {
	threeOnce.Do(func() {
		threeOnce.v = NewFloat(3.0)
	})
	return threeOnce.v
}

// Sqrt sets z to the rounded square root of x, and returns it.
//
// If z's precision is 0, it is changed to x's precision before the
// operation. Rounding is performed according to z's precision and
// rounding mode, but z's accuracy is not computed. Specifically, the
// result of z.Acc() is undefined.
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
		z.exp++
	case -1:
		z.exp--
	}
	// 0.25 <= z < 2.0

	// Solving 1/x² - z = 0 avoids Quo calls and is faster, especially
	// for high precisions.
	z.sqrtInverse(z)

	// re-attach halved exponent
	return z.SetMantExp(z, b/2)
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
	three := three()
	ng := func(t *Float) *Float {
		u.prec = t.prec
		v.prec = t.prec
		u.Mul(t, t)     // u = t²
		u.Mul(x, u)     //   = xt²
		v.Sub(three, u) // v = 3 - xt²
		u.Mul(t, v)     // u = t(3 - xt²)
		u.exp--         //   = ½t(3 - xt²)
		return t.Set(u)
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

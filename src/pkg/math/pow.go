// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math


// Pow returns x**y, the base-x exponential of y.
func Pow(x, y float64) float64 {
	// TODO: x or y NaN, ±Inf, maybe ±0.
	switch {
	case y == 0:
		return 1
	case y == 1:
		return x
	case x == 0 && y > 0:
		return 0
	case x == 0 && y < 0:
		return Inf(1)
	case y == 0.5:
		return Sqrt(x)
	case y == -0.5:
		return 1 / Sqrt(x)
	}

	absy := y;
	flip := false;
	if absy < 0 {
		absy = -absy;
		flip = true;
	}
	yi, yf := Modf(absy);
	if yf != 0 && x < 0 {
		return NaN()
	}
	if yi >= 1<<63 {
		return Exp(y * Log(x))
	}

	// ans = a1 * 2^ae (= 1 for now).
	a1 := float64(1);
	ae := 0;

	// ans *= x^yf
	if yf != 0 {
		if yf > 0.5 {
			yf--;
			yi++;
		}
		a1 = Exp(yf * Log(x));
	}

	// ans *= x^yi
	// by multiplying in successive squarings
	// of x according to bits of yi.
	// accumulate powers of two into exp.
	x1, xe := Frexp(x);
	for i := int64(yi); i != 0; i >>= 1 {
		if i&1 == 1 {
			a1 *= x1;
			ae += xe;
		}
		x1 *= x1;
		xe <<= 1;
		if x1 < .5 {
			x1 += x1;
			xe--;
		}
	}

	// ans = a1*2^ae
	// if flip { ans = 1 / ans }
	// but in the opposite order
	if flip {
		a1 = 1 / a1;
		ae = -ae;
	}
	return Ldexp(a1, ae);
}

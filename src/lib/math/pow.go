// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

import "math"

// x^y: exponentation
export func Pow(x, y float64) float64 {
	// TODO: x or y NaN, ±Inf, maybe ±0.
	switch {
	case y == 0:
		return 1;
	case y == 1:
		return x;
	case x == 0 && y > 0:
		return 0;
	case x == 0 && y < 0:
		return sys.Inf(1);
	case y == 0.5:
		return Sqrt(x);
	case y == -0.5:
		return 1 / Sqrt(x);
	}

	absy := y;
	flip := false;
	if absy < 0 {
		absy = -absy;
		flip = true;
	}
	yi, yf := sys.modf(absy);
	if yf != 0 && x < 0 {
		return sys.NaN();
	}
	if yi >= 1<<63 {
		return Exp(y * Log(x));
	}

	ans := float64(1);

	// ans *= x^yf
	if yf != 0 {
		if yf > 0.5 {
			yf--;
			yi++;
		}
		ans = Exp(yf * Log(x));
	}

	// ans *= x^yi
	// by multiplying in successive squarings
	// of x according to bits of yi.
	// accumulate powers of two into exp.
	// will still have to do ans *= 2^exp later.
	x1, xe := sys.frexp(x);
	exp := 0;
	if i := int64(yi); i != 0 {
		for {
			if i&1 == 1 {
				ans *= x1;
				exp += xe;
			}
			i >>= 1;
			if i == 0 {
				break;
			}
			x1 *= x1;
			xe <<= 1;
			if x1 < .5 {
				x1 += x1;
				xe--;
			}
		}
	}

	// ans *= 2^exp
	// if flip { ans = 1 / ans }
	// but in the opposite order
	if flip {
		ans = 1 / ans;
		exp = -exp;
	}
	return sys.ldexp(ans, exp);
}


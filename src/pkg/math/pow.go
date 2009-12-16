// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

func isOddInt(x float64) bool {
	xi, xf := Modf(x)
	return xf == 0 && int64(xi)&1 == 1
}

// Pow returns x**y, the base-x exponential of y.
func Pow(x, y float64) float64 {
	// TODO:  maybe ±0.
	// TODO(rsc): Remove manual inlining of IsNaN, IsInf
	// when compiler does it for us
	switch {
	case y == 0:
		return 1
	case y == 1:
		return x
	case y == 0.5:
		return Sqrt(x)
	case y == -0.5:
		return 1 / Sqrt(x)
	case x != x || y != y: // IsNaN(x) || IsNaN(y):
		return NaN()
	case x == 0:
		switch {
		case y < 0:
			return Inf(1)
		case y > 0:
			return 0
		}
	case y > MaxFloat64 || y < -MaxFloat64: // IsInf(y, 0):
		switch {
		case Fabs(x) == 1:
			return NaN()
		case Fabs(x) < 1:
			switch {
			case IsInf(y, -1):
				return Inf(1)
			case IsInf(y, 1):
				return 0
			}
		case Fabs(x) > 1:
			switch {
			case IsInf(y, -1):
				return 0
			case IsInf(y, 1):
				return Inf(1)
			}
		}
	case x > MaxFloat64 || x < -MaxFloat64: // IsInf(x, 0):
		switch {
		case y < 0:
			return 0
		case y > 0:
			switch {
			case IsInf(x, -1):
				if isOddInt(y) {
					return Inf(-1)
				}
				return Inf(1)
			case IsInf(x, 1):
				return Inf(1)
			}
		}
	}

	absy := y
	flip := false
	if absy < 0 {
		absy = -absy
		flip = true
	}
	yi, yf := Modf(absy)
	if yf != 0 && x < 0 {
		return NaN()
	}
	if yi >= 1<<63 {
		return Exp(y * Log(x))
	}

	// ans = a1 * 2^ae (= 1 for now).
	a1 := float64(1)
	ae := 0

	// ans *= x^yf
	if yf != 0 {
		if yf > 0.5 {
			yf--
			yi++
		}
		a1 = Exp(yf * Log(x))
	}

	// ans *= x^yi
	// by multiplying in successive squarings
	// of x according to bits of yi.
	// accumulate powers of two into exp.
	x1, xe := Frexp(x)
	for i := int64(yi); i != 0; i >>= 1 {
		if i&1 == 1 {
			a1 *= x1
			ae += xe
		}
		x1 *= x1
		xe <<= 1
		if x1 < .5 {
			x1 += x1
			xe--
		}
	}

	// ans = a1*2^ae
	// if flip { ans = 1 / ans }
	// but in the opposite order
	if flip {
		a1 = 1 / a1
		ae = -ae
	}
	return Ldexp(a1, ae)
}

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

/*
	Inverse of the floating-point error function.
*/

// This implementation is based on the rational approximation
// of percentage points of normal distribution available from
// https://www.jstor.org/stable/2347330.

const (
	// Coefficients for approximation to erf in |x| <= 0.85
	a0 = 1.1975323115670912564578e0
	a1 = 4.7072688112383978012285e1
	a2 = 6.9706266534389598238465e2
	a3 = 4.8548868893843886794648e3
	a4 = 1.6235862515167575384252e4
	a5 = 2.3782041382114385731252e4
	a6 = 1.1819493347062294404278e4
	a7 = 8.8709406962545514830200e2
	b0 = 1.0000000000000000000e0
	b1 = 4.2313330701600911252e1
	b2 = 6.8718700749205790830e2
	b3 = 5.3941960214247511077e3
	b4 = 2.1213794301586595867e4
	b5 = 3.9307895800092710610e4
	b6 = 2.8729085735721942674e4
	b7 = 5.2264952788528545610e3
	// Coefficients for approximation to erf in 0.85 < |x| <= 1-2*exp(-25)
	c0 = 1.42343711074968357734e0
	c1 = 4.63033784615654529590e0
	c2 = 5.76949722146069140550e0
	c3 = 3.64784832476320460504e0
	c4 = 1.27045825245236838258e0
	c5 = 2.41780725177450611770e-1
	c6 = 2.27238449892691845833e-2
	c7 = 7.74545014278341407640e-4
	d0 = 1.4142135623730950488016887e0
	d1 = 2.9036514445419946173133295e0
	d2 = 2.3707661626024532365971225e0
	d3 = 9.7547832001787427186894837e-1
	d4 = 2.0945065210512749128288442e-1
	d5 = 2.1494160384252876777097297e-2
	d6 = 7.7441459065157709165577218e-4
	d7 = 1.4859850019840355905497876e-9
	// Coefficients for approximation to erf in 1-2*exp(-25) < |x| < 1
	e0 = 6.65790464350110377720e0
	e1 = 5.46378491116411436990e0
	e2 = 1.78482653991729133580e0
	e3 = 2.96560571828504891230e-1
	e4 = 2.65321895265761230930e-2
	e5 = 1.24266094738807843860e-3
	e6 = 2.71155556874348757815e-5
	e7 = 2.01033439929228813265e-7
	f0 = 1.414213562373095048801689e0
	f1 = 8.482908416595164588112026e-1
	f2 = 1.936480946950659106176712e-1
	f3 = 2.103693768272068968719679e-2
	f4 = 1.112800997078859844711555e-3
	f5 = 2.611088405080593625138020e-5
	f6 = 2.010321207683943062279931e-7
	f7 = 2.891024605872965461538222e-15
)

// Erfinv returns the inverse error function of x.
//
// Special cases are:
//
//	Erfinv(1) = +Inf
//	Erfinv(-1) = -Inf
//	Erfinv(x) = NaN if x < -1 or x > 1
//	Erfinv(NaN) = NaN
func Erfinv(x float64) float64 {
	// special cases
	if IsNaN(x) || x <= -1 || x >= 1 {
		if x == -1 || x == 1 {
			return Inf(int(x))
		}
		return NaN()
	}

	sign := false
	if x < 0 {
		x = -x
		sign = true
	}

	var ans float64
	if x <= 0.85 { // |x| <= 0.85
		r := 0.180625 - 0.25*x*x
		z1 := ((((((a7*r+a6)*r+a5)*r+a4)*r+a3)*r+a2)*r+a1)*r + a0
		z2 := ((((((b7*r+b6)*r+b5)*r+b4)*r+b3)*r+b2)*r+b1)*r + b0
		ans = (x * z1) / z2
	} else {
		var z1, z2 float64
		r := Sqrt(Ln2 - Log(1.0-x))
		if r <= 5.0 {
			r -= 1.6
			z1 = ((((((c7*r+c6)*r+c5)*r+c4)*r+c3)*r+c2)*r+c1)*r + c0
			z2 = ((((((d7*r+d6)*r+d5)*r+d4)*r+d3)*r+d2)*r+d1)*r + d0
		} else {
			r -= 5.0
			z1 = ((((((e7*r+e6)*r+e5)*r+e4)*r+e3)*r+e2)*r+e1)*r + e0
			z2 = ((((((f7*r+f6)*r+f5)*r+f4)*r+f3)*r+f2)*r+f1)*r + f0
		}
		ans = z1 / z2
	}

	if sign {
		return -ans
	}
	return ans
}

// Erfcinv returns the inverse of [Erfc](x).
//
// Special cases are:
//
//	Erfcinv(0) = +Inf
//	Erfcinv(2) = -Inf
//	Erfcinv(x) = NaN if x < 0 or x > 2
//	Erfcinv(NaN) = NaN
func Erfcinv(x float64) float64 {
	return Erfinv(1 - x)
}

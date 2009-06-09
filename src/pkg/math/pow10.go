// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

/*
 * this table might overflow 127-bit exponent representations.
 * in that case, truncate it after 1.0e38.
 * it is important to get all one can from this
 * routine since it is used in atof to scale numbers.
 * the presumption is that GO converts fp numbers better
 * than multipication of lower powers of 10.
 */

var	pow10tab	[70]float64;

// Pow10 returns 10**x, the base-10 exponential of x.
func Pow10(e int) float64 {
	if e < 0 {
		return 1/Pow10(-e);
	}
	if e < len(pow10tab) {
		return pow10tab[e];
	}
	m := e/2;
	return Pow10(m) * Pow10(e-m);
}

func init() {
	pow10tab[0] = 1.0e0;
	pow10tab[1] = 1.0e1;
	for i:=2; i<len(pow10tab); i++ {
		m := i/2;
		pow10tab[i] = pow10tab[m] * pow10tab[i-m];
	}
}

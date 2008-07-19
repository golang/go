// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

export	pow10

/*
 * this table might overflow 127-bit exponent representations.
 * in that case, truncate it after 1.0e38.
 * it is important to get all one can from this
 * routine since it is used in atof to scale numbers.
 * the presumption is that GO converts fp numbers better
 * than multipication of lower powers of 10.
 */

const	tabsize		= 70;
var	tab[tabsize]	float64;

func
pow10(e int) float64 
{
	if e < 0 {
		return 1/pow10(-e);
	}
	if e < tabsize {
		return tab[e];
	}
	m := e/2;
	return pow10(m) * pow10(e-m);
}

func
init()
{
	tab[0] = 1.0e0;
	tab[1] = 1.0e1;
	for i:=2; i<tabsize; i++ {
		m := i/2;
		tab[i] = tab[m] * tab[i-m];
	}
}

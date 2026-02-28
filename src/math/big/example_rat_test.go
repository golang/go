// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big_test

import (
	"fmt"
	"math/big"
)

// Use the classic continued fraction for e
//
//	e = [1; 0, 1, 1, 2, 1, 1, ... 2n, 1, 1, ...]
//
// i.e., for the nth term, use
//
//	   1          if   n mod 3 != 1
//	(n-1)/3 * 2   if   n mod 3 == 1
func recur(n, lim int64) *big.Rat {
	term := new(big.Rat)
	if n%3 != 1 {
		term.SetInt64(1)
	} else {
		term.SetInt64((n - 1) / 3 * 2)
	}

	if n > lim {
		return term
	}

	// Directly initialize frac as the fractional
	// inverse of the result of recur.
	frac := new(big.Rat).Inv(recur(n+1, lim))

	return term.Add(term, frac)
}

// This example demonstrates how to use big.Rat to compute the
// first 15 terms in the sequence of rational convergents for
// the constant e (base of natural logarithm).
func Example_eConvergents() {
	for i := 1; i <= 15; i++ {
		r := recur(0, int64(i))

		// Print r both as a fraction and as a floating-point number.
		// Since big.Rat implements fmt.Formatter, we can use %-13s to
		// get a left-aligned string representation of the fraction.
		fmt.Printf("%-13s = %s\n", r, r.FloatString(8))
	}

	// Output:
	// 2/1           = 2.00000000
	// 3/1           = 3.00000000
	// 8/3           = 2.66666667
	// 11/4          = 2.75000000
	// 19/7          = 2.71428571
	// 87/32         = 2.71875000
	// 106/39        = 2.71794872
	// 193/71        = 2.71830986
	// 1264/465      = 2.71827957
	// 1457/536      = 2.71828358
	// 2721/1001     = 2.71828172
	// 23225/8544    = 2.71828184
	// 25946/9545    = 2.71828182
	// 49171/18089   = 2.71828183
	// 517656/190435 = 2.71828183
}

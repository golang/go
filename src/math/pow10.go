// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

// pow10tab stores the pre-computed values 10**i for i < 32.
var pow10tab = [...]float64{
	1e00, 1e01, 1e02, 1e03, 1e04, 1e05, 1e06, 1e07, 1e08, 1e09,
	1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
	1e20, 1e21, 1e22, 1e23, 1e24, 1e25, 1e26, 1e27, 1e28, 1e29,
	1e30, 1e31,
}

// pow10postab32 stores the pre-computed value for 10**(i*32) at index i.
var pow10postab32 = [...]float64{
	1e00, 1e32, 1e64, 1e96, 1e128, 1e160, 1e192, 1e224, 1e256, 1e288,
}

// pow10negtab32 stores the pre-computed value for 10**(-i*32) at index i.
var pow10negtab32 = [...]float64{
	1e-00, 1e-32, 1e-64, 1e-96, 1e-128, 1e-160, 1e-192, 1e-224, 1e-256, 1e-288, 1e-320,
}

// Pow10 returns 10**n, the base-10 exponential of n.
//
// Special cases are:
//	Pow10(n) =    0 for n < -323
//	Pow10(n) = +Inf for n > 308
func Pow10(n int) float64 {
	if 0 <= n && n <= 308 {
		return pow10postab32[uint(n)/32] * pow10tab[uint(n)%32]
	}

	if -323 <= n && n <= 0 {
		return pow10negtab32[uint(-n)/32] / pow10tab[uint(-n)%32]
	}

	// n < -323 || 308 < n
	if n > 0 {
		return Inf(1)
	}

	// n < -323
	return 0
}

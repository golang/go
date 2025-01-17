// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Multiplication.

package big

// Operands that are shorter than karatsubaThreshold are multiplied using
// "grade school" multiplication; for longer operands the Karatsuba algorithm
// is used.
var karatsubaThreshold = 40 // computed by calibrate_test.go

// mul sets z = x*y, using stk for temporary storage.
// The caller may pass stk == nil to request that mul obtain and release one itself.
func (z nat) mul(stk *stack, x, y nat) nat {
	m := len(x)
	n := len(y)

	switch {
	case m < n:
		return z.mul(stk, y, x)
	case m == 0 || n == 0:
		return z[:0]
	case n == 1:
		return z.mulAddWW(x, y[0], 0)
	}
	// m >= n > 1

	// determine if z can be reused
	if alias(z, x) || alias(z, y) {
		z = nil // z is an alias for x or y - cannot reuse
	}

	// use basic multiplication if the numbers are small
	if n < karatsubaThreshold {
		z = z.make(m + n)
		basicMul(z, x, y)
		return z.norm()
	}
	// m >= n && n >= karatsubaThreshold && n >= 2

	// determine Karatsuba length k such that
	//
	//   x = xh*b + x0  (0 <= x0 < b)
	//   y = yh*b + y0  (0 <= y0 < b)
	//   b = 1<<(_W*k)  ("base" of digits xi, yi)
	//
	k := karatsubaLen(n, karatsubaThreshold)
	// k <= n

	if stk == nil {
		stk = getStack()
		defer stk.free()
	}

	// multiply x0 and y0 via Karatsuba
	x0 := x[0:k]      // x0 is not normalized
	y0 := y[0:k]      // y0 is not normalized
	z = z.make(m + n) // enough space for full result of x*y
	karatsuba(stk, z, x0, y0)
	clear(z[2*k:]) // upper portion of z is garbage (and 2*k <= m+n since k <= n <= m)

	// If xh != 0 or yh != 0, add the missing terms to z. For
	//
	//   xh = xi*b^i + ... + x2*b^2 + x1*b (0 <= xi < b)
	//   yh =                         y1*b (0 <= y1 < b)
	//
	// the missing terms are
	//
	//   x0*y1*b and xi*y0*b^i, xi*y1*b^(i+1) for i > 0
	//
	// since all the yi for i > 1 are 0 by choice of k: If any of them
	// were > 0, then yh >= b^2 and thus y >= b^2. Then k' = k*2 would
	// be a larger valid threshold contradicting the assumption about k.
	//
	if k < n || m != n {
		defer stk.restore(stk.save())
		t := stk.nat(3 * k)

		// add x0*y1*b
		x0 := x0.norm()
		y1 := y[k:]            // y1 is normalized because y is
		t = t.mul(stk, x0, y1) // update t so we don't lose t's underlying array
		addAt(z, t, k)

		// add xi*y0<<i, xi*y1*b<<(i+k)
		y0 := y0.norm()
		for i := k; i < len(x); i += k {
			xi := x[i:]
			if len(xi) > k {
				xi = xi[:k]
			}
			xi = xi.norm()
			t = t.mul(stk, xi, y0)
			addAt(z, t, i)
			t = t.mul(stk, xi, y1)
			addAt(z, t, i+k)
		}
	}

	return z.norm()
}

// Operands that are shorter than basicSqrThreshold are squared using
// "grade school" multiplication; for operands longer than karatsubaSqrThreshold
// we use the Karatsuba algorithm optimized for x == y.
var basicSqrThreshold = 20      // computed by calibrate_test.go
var karatsubaSqrThreshold = 260 // computed by calibrate_test.go

// sqr sets z = x*x, using stk for temporary storage.
// The caller may pass stk == nil to request that sqr obtain and release one itself.
func (z nat) sqr(stk *stack, x nat) nat {
	n := len(x)
	switch {
	case n == 0:
		return z[:0]
	case n == 1:
		d := x[0]
		z = z.make(2)
		z[1], z[0] = mulWW(d, d)
		return z.norm()
	}

	if alias(z, x) {
		z = nil // z is an alias for x - cannot reuse
	}
	z = z.make(2 * n)

	if n < basicSqrThreshold {
		basicMul(z, x, x)
		return z.norm()
	}

	if stk == nil {
		stk = getStack()
		defer stk.free()
	}

	if n < karatsubaSqrThreshold {
		basicSqr(stk, z, x)
		return z.norm()
	}

	// Use Karatsuba multiplication optimized for x == y.
	// The algorithm and layout of z are the same as for mul.

	// z = (x1*b + x0)^2 = x1^2*b^2 + 2*x1*x0*b + x0^2

	k := karatsubaLen(n, karatsubaSqrThreshold)

	x0 := x[0:k]
	karatsubaSqr(stk, z, x0) // z = x0^2
	clear(z[2*k:])

	if k < n {
		t := stk.nat(2 * k)
		x0 := x0.norm()
		x1 := x[k:]
		t = t.mul(stk, x0, x1)
		addAt(z, t, k)
		addAt(z, t, k) // z = 2*x1*x0*b + x0^2
		t = t.sqr(stk, x1)
		addAt(z, t, 2*k) // z = x1^2*b^2 + 2*x1*x0*b + x0^2
	}

	return z.norm()
}

// basicSqr sets z = x*x and is asymptotically faster than basicMul
// by about a factor of 2, but slower for small arguments due to overhead.
// Requirements: len(x) > 0, len(z) == 2*len(x)
// The (non-normalized) result is placed in z.
func basicSqr(stk *stack, z, x nat) {
	n := len(x)
	defer stk.restore(stk.save())
	t := stk.nat(2 * n)
	clear(t)
	z[1], z[0] = mulWW(x[0], x[0]) // the initial square
	for i := 1; i < n; i++ {
		d := x[i]
		// z collects the squares x[i] * x[i]
		z[2*i+1], z[2*i] = mulWW(d, d)
		// t collects the products x[i] * x[j] where j < i
		t[2*i] = addMulVVW(t[i:2*i], x[0:i], d)
	}
	t[2*n-1] = shlVU(t[1:2*n-1], t[1:2*n-1], 1) // double the j < i products
	addVV(z, z, t)                              // combine the result
}

func (z nat) mulAddWW(x nat, y, r Word) nat {
	m := len(x)
	if m == 0 || y == 0 {
		return z.setWord(r) // result is r
	}
	// m > 0

	z = z.make(m + 1)
	z[m] = mulAddVWW(z[0:m], x, y, r)

	return z.norm()
}

// basicMul multiplies x and y and leaves the result in z.
// The (non-normalized) result is placed in z[0 : len(x) + len(y)].
func basicMul(z, x, y nat) {
	clear(z[0 : len(x)+len(y)]) // initialize z
	for i, d := range y {
		if d != 0 {
			z[len(x)+i] = addMulVVW(z[i:i+len(x)], x, d)
		}
	}
}

// karatsubaLen computes an approximation to the maximum k <= n such that
// k = p<<i for a number p <= threshold and an i >= 0. Thus, the
// result is the largest number that can be divided repeatedly by 2 before
// becoming about the value of threshold.
func karatsubaLen(n, threshold int) int {
	i := uint(0)
	for n > threshold {
		n >>= 1
		i++
	}
	return n << i
}

// Fast version of z[0:n+n>>1].add(z[0:n+n>>1], x[0:n]) w/o bounds checks.
// Factored out for readability - do not use outside karatsuba.
func karatsubaAdd(z, x nat, n int) {
	if c := addVV(z[0:n], z, x); c != 0 {
		addVW(z[n:n+n>>1], z[n:], c)
	}
}

// Like karatsubaAdd, but does subtract.
func karatsubaSub(z, x nat, n int) {
	if c := subVV(z[0:n], z, x); c != 0 {
		subVW(z[n:n+n>>1], z[n:], c)
	}
}

// karatsuba multiplies x and y and leaves the result in z.
// Both x and y must have the same length n and n must be a
// power of 2. The result vector z must have len(z) == len(x)+len(y).
// The (non-normalized) result is placed in z.
func karatsuba(stk *stack, z, x, y nat) {
	n := len(y)

	// Switch to basic multiplication if numbers are odd or small.
	// (n is always even if karatsubaThreshold is even, but be
	// conservative)
	if n&1 != 0 || n < karatsubaThreshold || n < 2 {
		basicMul(z, x, y)
		return
	}
	// n&1 == 0 && n >= karatsubaThreshold && n >= 2

	// Karatsuba multiplication is based on the observation that
	// for two numbers x and y with:
	//
	//   x = x1*b + x0
	//   y = y1*b + y0
	//
	// the product x*y can be obtained with 3 products z2, z1, z0
	// instead of 4:
	//
	//   x*y = x1*y1*b*b + (x1*y0 + x0*y1)*b + x0*y0
	//       =    z2*b*b +              z1*b +    z0
	//
	// with:
	//
	//   xd = x1 - x0
	//   yd = y0 - y1
	//
	//   z1 =      xd*yd                    + z2 + z0
	//      = (x1-x0)*(y0 - y1)             + z2 + z0
	//      = x1*y0 - x1*y1 - x0*y0 + x0*y1 + z2 + z0
	//      = x1*y0 -    z2 -    z0 + x0*y1 + z2 + z0
	//      = x1*y0                 + x0*y1

	// split x, y into "digits"
	n2 := n >> 1              // n2 >= 1
	x1, x0 := x[n2:], x[0:n2] // x = x1*b + y0
	y1, y0 := y[n2:], y[0:n2] // y = y1*b + y0

	// compute z0 and z2 with the result "in place" in z
	karatsuba(stk, z, x0, y0)     // z0 = x0*y0
	karatsuba(stk, z[n:], x1, y1) // z2 = x1*y1

	// compute xd, yd (or the negative value if underflow occurs)
	s := 1 // sign of product xd*yd
	defer stk.restore(stk.save())
	xd := stk.nat(n2)
	yd := stk.nat(n2)
	if subVV(xd, x1, x0) != 0 { // x1-x0
		s = -s
		subVV(xd, x0, x1) // x0-x1
	}
	if subVV(yd, y0, y1) != 0 { // y0-y1
		s = -s
		subVV(yd, y1, y0) // y1-y0
	}

	// p = (x1-x0)*(y0-y1) == x1*y0 - x1*y1 - x0*y0 + x0*y1 for s > 0
	// p = (x0-x1)*(y0-y1) == x0*y0 - x0*y1 - x1*y0 + x1*y1 for s < 0
	p := stk.nat(2 * n2)
	karatsuba(stk, p, xd, yd)

	// save original z2:z0
	// (ok to use upper half of z since we're done recurring)
	r := stk.nat(n * 2)
	copy(r, z[:n*2])

	// add up all partial products
	//
	//   2*n     n     0
	// z = [ z2  | z0  ]
	//   +    [ z0  ]
	//   +    [ z2  ]
	//   +    [  p  ]
	//
	karatsubaAdd(z[n2:], r, n)
	karatsubaAdd(z[n2:], r[n:], n)
	if s > 0 {
		karatsubaAdd(z[n2:], p, n)
	} else {
		karatsubaSub(z[n2:], p, n)
	}
}

// karatsubaSqr squares x and leaves the result in z.
// len(x) must be a power of 2 and len(z) == 2*len(x).
// The (non-normalized) result is placed in z.
//
// The algorithm and the layout of z are the same as for karatsuba.
func karatsubaSqr(stk *stack, z, x nat) {
	n := len(x)

	if n&1 != 0 || n < karatsubaSqrThreshold || n < 2 {
		basicSqr(stk, z[:2*n], x)
		return
	}

	n2 := n >> 1
	x1, x0 := x[n2:], x[0:n2]

	karatsubaSqr(stk, z, x0)
	karatsubaSqr(stk, z[n:], x1)

	// s = sign(xd*yd) == -1 for xd != 0; s == 1 for xd == 0
	defer stk.restore(stk.save())
	p := stk.nat(2 * n2)
	r := stk.nat(n * 2)
	xd := r[:n2]
	if subVV(xd, x1, x0) != 0 {
		subVV(xd, x0, x1)
	}

	karatsubaSqr(stk, p, xd)
	copy(r, z[:n*2])

	karatsubaAdd(z[n2:], r, n)
	karatsubaAdd(z[n2:], r[n:], n)
	karatsubaSub(z[n2:], p, n) // s == -1 for p != 0; s == 1 for p == 0
}

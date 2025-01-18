// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Multiplication.

package big

// Operands that are shorter than karatsubaThreshold are multiplied using
// "grade school" multiplication; for longer operands the Karatsuba algorithm
// is used.
var karatsubaThreshold = 40 // see calibrate_test.go

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
	z = z.make(m + n)

	// use basic multiplication if the numbers are small
	if n < karatsubaThreshold {
		basicMul(z, x, y)
		return z.norm()
	}

	if stk == nil {
		stk = getStack()
		defer stk.free()
	}

	// Let x = x1:x0 where x0 is the same length as y.
	// Compute z = x0*y and then add in x1*y in sections
	// if needed.
	karatsuba(stk, z[:2*n], x[:n], y)

	if n < m {
		clear(z[2*n:])
		defer stk.restore(stk.save())
		t := stk.nat(2 * n)
		for i := n; i < m; i += n {
			t = t.mul(stk, x[i:min(i+n, len(x))], y)
			addTo(z[i:], t)
		}
	}

	return z.norm()
}

// Operands that are shorter than basicSqrThreshold are squared using
// "grade school" multiplication; for operands longer than karatsubaSqrThreshold
// we use the Karatsuba algorithm optimized for x == y.
var basicSqrThreshold = 12     // see calibrate_test.go
var karatsubaSqrThreshold = 80 // see calibrate_test.go

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

	if n < basicSqrThreshold && n < karatsubaSqrThreshold {
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

	karatsubaSqr(stk, z, x)
	return z.norm()
}

// basicSqr sets z = x*x and is asymptotically faster than basicMul
// by about a factor of 2, but slower for small arguments due to overhead.
// Requirements: len(x) > 0, len(z) == 2*len(x)
// The (non-normalized) result is placed in z.
func basicSqr(stk *stack, z, x nat) {
	n := len(x)
	if n < basicSqrThreshold {
		basicMul(z, x, x)
		return
	}

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

// mulAddWW returns z = x*y + r.
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

// karatsuba multiplies x and y,
// writing the (non-normalized) result to z.
// x and y must have the same length n,
// and z must have length twice that.
func karatsuba(stk *stack, z, x, y nat) {
	n := len(y)
	if len(x) != n || len(z) != 2*n {
		panic("bad karatsuba length")
	}

	// Fall back to basic algorithm if small enough.
	if n < karatsubaThreshold || n < 2 {
		basicMul(z, x, y)
		return
	}

	// Let the notation x1:x0 denote the nat (x1<<N)+x0 for some N,
	// and similarly z2:z1:z0 = (z2<<2N)+(z1<<N)+z0.
	//
	// (Note that z0, z1, z2 might be ≥ 2**N, in which case the high
	// bits of, say, z0 are being added to the low bits of z1 in this notation.)
	//
	// Karatsuba multiplication is based on the observation that
	//
	//	x1:x0 * y1:y0 = x1*y1:(x0*y1+y0*x1):x0*y0
	//	              = x1*y1:((x0-x1)*(y1-y0)+x1*y1+x0*y0):x0*y0
	//
	// The second form uses only three half-width multiplications
	// instead of the four that the straightforward first form does.
	//
	// We call the three pieces z0, z1, z2:
	//
	//	z0 = x0*y0
	//	z2 = x1*y1
	//	z1 = (x0-x1)*(y1-y0) + z0 + z2

	n2 := (n + 1) / 2
	x0, x1 := &Int{abs: x[:n2].norm()}, &Int{abs: x[n2:].norm()}
	y0, y1 := &Int{abs: y[:n2].norm()}, &Int{abs: y[n2:].norm()}
	z0 := &Int{abs: z[0 : 2*n2]}
	z2 := &Int{abs: z[2*n2:]}

	// Allocate temporary storage for z1; repurpose z0 to hold tx and ty.
	defer stk.restore(stk.save())
	z1 := &Int{abs: stk.nat(2*n2 + 1)}
	tx := &Int{abs: z[0:n2]}
	ty := &Int{abs: z[n2 : 2*n2]}

	tx.Sub(x0, x1)
	ty.Sub(y1, y0)
	z1.mul(stk, tx, ty)

	clear(z)
	z0.mul(stk, x0, y0)
	z2.mul(stk, x1, y1)
	z1.Add(z1, z0)
	z1.Add(z1, z2)
	addTo(z[n2:], z1.abs)

	// Debug mode: double-check answer and print trace on failure.
	const debug = false
	if debug {
		zz := make(nat, len(z))
		basicMul(zz, x, y)
		if z.cmp(zz) != 0 {
			// All the temps were aliased to z and gone. Recompute.
			z0 = new(Int)
			z0.mul(stk, x0, y0)
			tx = new(Int).Sub(x1, x0)
			ty = new(Int).Sub(y0, y1)
			z2 = new(Int)
			z2.mul(stk, x1, y1)
			print("karatsuba wrong\n")
			trace("x ", &Int{abs: x})
			trace("y ", &Int{abs: y})
			trace("z ", &Int{abs: z})
			trace("zz", &Int{abs: zz})
			trace("x0", x0)
			trace("x1", x1)
			trace("y0", y0)
			trace("y1", y1)
			trace("tx", tx)
			trace("ty", ty)
			trace("z0", z0)
			trace("z1", z1)
			trace("z2", z2)
			panic("karatsuba")
		}
	}

}

// karatsubaSqr squares x,
// writing the (non-normalized) result to z.
// z must have length 2*len(x).
// It is analogous to [karatsuba] but can run faster
// knowing both multiplicands are the same value.
func karatsubaSqr(stk *stack, z, x nat) {
	n := len(x)
	if len(z) != 2*n {
		panic("bad karatsubaSqr length")
	}

	if n < karatsubaSqrThreshold || n < 2 {
		basicSqr(stk, z, x)
		return
	}

	// Recall that for karatsuba we want to compute:
	//
	//	x1:x0 * y1:y0 = x1y1:(x0y1+y0x1):x0y0
	//                = x1y1:((x0-x1)*(y1-y0)+x1y1+x0y0):x0y0
	//	              = z2:z1:z0
	// where:
	//
	//	z0 = x0y0
	//	z2 = x1y1
	//	z1 = (x0-x1)*(y1-y0) + z0 + z2
	//
	// When x = y, these simplify to:
	//
	//	z0 = x0²
	//	z2 = x1²
	//	z1 = z0 + z2 - (x0-x1)²

	n2 := (n + 1) / 2
	x0, x1 := &Int{abs: x[:n2].norm()}, &Int{abs: x[n2:].norm()}
	z0 := &Int{abs: z[0 : 2*n2]}
	z2 := &Int{abs: z[2*n2:]}

	// Allocate temporary storage for z1; repurpose z0 to hold tx.
	defer stk.restore(stk.save())
	z1 := &Int{abs: stk.nat(2*n2 + 1)}
	tx := &Int{abs: z[0:n2]}

	tx.Sub(x0, x1)
	z1.abs = z1.abs.sqr(stk, tx.abs)
	z1.neg = true

	clear(z)
	z0.abs = z0.abs.sqr(stk, x0.abs)
	z2.abs = z2.abs.sqr(stk, x1.abs)
	z1.Add(z1, z0)
	z1.Add(z1, z2)
	addTo(z[n2:], z1.abs)

	// Debug mode: double-check answer and print trace on failure.
	const debug = false
	if debug {
		zz := make(nat, len(z))
		basicSqr(stk, zz, x)
		if z.cmp(zz) != 0 {
			// All the temps were aliased to z and gone. Recompute.
			tx = new(Int).Sub(x0, x1)
			z0 = new(Int).Mul(x0, x0)
			z2 = new(Int).Mul(x1, x1)
			z1 = new(Int).Mul(tx, tx)
			z1.Neg(z1)
			z1.Add(z1, z0)
			z1.Add(z1, z2)
			print("karatsubaSqr wrong\n")
			trace("x ", &Int{abs: x})
			trace("z ", &Int{abs: z})
			trace("zz", &Int{abs: zz})
			trace("x0", x0)
			trace("x1", x1)
			trace("z0", z0)
			trace("z1", z1)
			trace("z2", z2)
			panic("karatsubaSqr")
		}
	}
}

// ifmt returns the debug formatting of the Int x: 0xHEX.
func ifmt(x *Int) string {
	neg, s, t := "", x.Text(16), ""
	if s == "" { // happens for denormalized zero
		s = "0x0"
	}
	if s[0] == '-' {
		neg, s = "-", s[1:]
	}

	// Add _ between words.
	const D = _W / 4 // digits per chunk
	for len(s) > D {
		s, t = s[:len(s)-D], s[len(s)-D:]+"_"+t
	}
	return neg + s + t
}

// trace prints a single debug value.
func trace(name string, x *Int) {
	print(name, "=", ifmt(x), "\n")
}

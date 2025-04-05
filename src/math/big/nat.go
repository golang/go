// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements unsigned multi-precision integers (natural
// numbers). They are the building blocks for the implementation
// of signed integers, rationals, and floating-point numbers.
//
// Caution: This implementation relies on the function "alias"
//          which assumes that (nat) slice capacities are never
//          changed (no 3-operand slice expressions). If that
//          changes, alias needs to be updated for correctness.

package big

import (
	"internal/byteorder"
	"math/bits"
	"math/rand"
	"slices"
	"sync"
)

// An unsigned integer x of the form
//
//	x = x[n-1]*_B^(n-1) + x[n-2]*_B^(n-2) + ... + x[1]*_B + x[0]
//
// with 0 <= x[i] < _B and 0 <= i < n is stored in a slice of length n,
// with the digits x[i] as the slice elements.
//
// A number is normalized if the slice contains no leading 0 digits.
// During arithmetic operations, denormalized values may occur but are
// always normalized before returning the final result. The normalized
// representation of 0 is the empty or nil slice (length = 0).
type nat []Word

var (
	natOne  = nat{1}
	natTwo  = nat{2}
	natFive = nat{5}
	natTen  = nat{10}
)

func (z nat) String() string {
	return "0x" + string(z.itoa(false, 16))
}

func (z nat) norm() nat {
	i := len(z)
	for i > 0 && z[i-1] == 0 {
		i--
	}
	return z[0:i]
}

func (z nat) make(n int) nat {
	if n <= cap(z) {
		return z[:n] // reuse z
	}
	if n == 1 {
		// Most nats start small and stay that way; don't over-allocate.
		return make(nat, 1)
	}
	// Choosing a good value for e has significant performance impact
	// because it increases the chance that a value can be reused.
	const e = 4 // extra capacity
	return make(nat, n, n+e)
}

func (z nat) setWord(x Word) nat {
	if x == 0 {
		return z[:0]
	}
	z = z.make(1)
	z[0] = x
	return z
}

func (z nat) setUint64(x uint64) nat {
	// single-word value
	if w := Word(x); uint64(w) == x {
		return z.setWord(w)
	}
	// 2-word value
	z = z.make(2)
	z[1] = Word(x >> 32)
	z[0] = Word(x)
	return z
}

func (z nat) set(x nat) nat {
	z = z.make(len(x))
	copy(z, x)
	return z
}

func (z nat) add(x, y nat) nat {
	m := len(x)
	n := len(y)

	switch {
	case m < n:
		return z.add(y, x)
	case m == 0:
		// n == 0 because m >= n; result is 0
		return z[:0]
	case n == 0:
		// result is x
		return z.set(x)
	}
	// m > 0

	z = z.make(m + 1)
	c := addVV(z[0:n], x, y)
	if m > n {
		c = addVW(z[n:m], x[n:], c)
	}
	z[m] = c

	return z.norm()
}

func (z nat) sub(x, y nat) nat {
	m := len(x)
	n := len(y)

	switch {
	case m < n:
		panic("underflow")
	case m == 0:
		// n == 0 because m >= n; result is 0
		return z[:0]
	case n == 0:
		// result is x
		return z.set(x)
	}
	// m > 0

	z = z.make(m)
	c := subVV(z[0:n], x, y)
	if m > n {
		c = subVW(z[n:], x[n:], c)
	}
	if c != 0 {
		panic("underflow")
	}

	return z.norm()
}

func (x nat) cmp(y nat) (r int) {
	m := len(x)
	n := len(y)
	if m != n || m == 0 {
		switch {
		case m < n:
			r = -1
		case m > n:
			r = 1
		}
		return
	}

	i := m - 1
	for i > 0 && x[i] == y[i] {
		i--
	}

	switch {
	case x[i] < y[i]:
		r = -1
	case x[i] > y[i]:
		r = 1
	}
	return
}

// montgomery computes z mod m = x*y*2**(-n*_W) mod m,
// assuming k = -1/m mod 2**_W.
// z is used for storing the result which is returned;
// z must not alias x, y or m.
// See Gueron, "Efficient Software Implementations of Modular Exponentiation".
// https://eprint.iacr.org/2011/239.pdf
// In the terminology of that paper, this is an "Almost Montgomery Multiplication":
// x and y are required to satisfy 0 <= z < 2**(n*_W) and then the result
// z is guaranteed to satisfy 0 <= z < 2**(n*_W), but it may not be < m.
func (z nat) montgomery(x, y, m nat, k Word, n int) nat {
	// This code assumes x, y, m are all the same length, n.
	// (required by addMulVVW and the for loop).
	// It also assumes that x, y are already reduced mod m,
	// or else the result will not be properly reduced.
	if len(x) != n || len(y) != n || len(m) != n {
		panic("math/big: mismatched montgomery number lengths")
	}
	z = z.make(n * 2)
	clear(z)
	var c Word
	for i := 0; i < n; i++ {
		d := y[i]
		c2 := addMulVVWW(z[i:n+i], z[i:n+i], x, d, 0)
		t := z[i] * k
		c3 := addMulVVWW(z[i:n+i], z[i:n+i], m, t, 0)
		cx := c + c2
		cy := cx + c3
		z[n+i] = cy
		if cx < c2 || cy < c3 {
			c = 1
		} else {
			c = 0
		}
	}
	if c != 0 {
		subVV(z[:n], z[n:], m)
	} else {
		copy(z[:n], z[n:])
	}
	return z[:n]
}

// alias reports whether x and y share the same base array.
//
// Note: alias assumes that the capacity of underlying arrays
// is never changed for nat values; i.e. that there are
// no 3-operand slice expressions in this code (or worse,
// reflect-based operations to the same effect).
func alias(x, y nat) bool {
	return cap(x) > 0 && cap(y) > 0 && &x[0:cap(x)][cap(x)-1] == &y[0:cap(y)][cap(y)-1]
}

// addTo implements z += x; z must be long enough.
// (we don't use nat.add because we need z to stay the same
// slice, and we don't need to normalize z after each addition)
func addTo(z, x nat) {
	if n := len(x); n > 0 {
		if c := addVV(z[:n], z, x); c != 0 {
			if n < len(z) {
				addVW(z[n:], z[n:], c)
			}
		}
	}
}

// mulRange computes the product of all the unsigned integers in the
// range [a, b] inclusively. If a > b (empty range), the result is 1.
// The caller may pass stk == nil to request that mulRange obtain and release one itself.
func (z nat) mulRange(stk *stack, a, b uint64) nat {
	switch {
	case a == 0:
		// cut long ranges short (optimization)
		return z.setUint64(0)
	case a > b:
		return z.setUint64(1)
	case a == b:
		return z.setUint64(a)
	case a+1 == b:
		return z.mul(stk, nat(nil).setUint64(a), nat(nil).setUint64(b))
	}

	if stk == nil {
		stk = getStack()
		defer stk.free()
	}

	m := a + (b-a)/2 // avoid overflow
	return z.mul(stk, nat(nil).mulRange(stk, a, m), nat(nil).mulRange(stk, m+1, b))
}

// A stack provides temporary storage for complex calculations
// such as multiplication and division.
// The stack is a simple slice of words, extended as needed
// to hold all the temporary storage for a calculation.
// In general, if a function takes a *stack, it expects a non-nil *stack.
// However, certain functions may allow passing a nil *stack instead,
// so that they can handle trivial stack-free cases without forcing the
// caller to obtain and free a stack that will be unused. These functions
// document that they accept a nil *stack in their doc comments.
type stack struct {
	w []Word
}

var stackPool sync.Pool

// getStack returns a temporary stack.
// The caller must call [stack.free] to give up use of the stack when finished.
func getStack() *stack {
	s, _ := stackPool.Get().(*stack)
	if s == nil {
		s = new(stack)
	}
	return s
}

// free returns the stack for use by another calculation.
func (s *stack) free() {
	s.w = s.w[:0]
	stackPool.Put(s)
}

// save returns the current stack pointer.
// A future call to restore with the same value
// frees any temporaries allocated on the stack after the call to save.
func (s *stack) save() int {
	return len(s.w)
}

// restore restores the stack pointer to n.
// It is almost always invoked as
//
//	defer stk.restore(stk.save())
//
// which makes sure to pop any temporaries allocated in the current function
// from the stack before returning.
func (s *stack) restore(n int) {
	s.w = s.w[:n]
}

// nat returns a nat of n words, allocated on the stack.
func (s *stack) nat(n int) nat {
	nr := (n + 3) &^ 3 // round up to multiple of 4
	off := len(s.w)
	s.w = slices.Grow(s.w, nr)
	s.w = s.w[:off+nr]
	x := s.w[off : off+n : off+n]
	if n > 0 {
		x[0] = 0xfedcb
	}
	return x
}

// bitLen returns the length of x in bits.
// Unlike most methods, it works even if x is not normalized.
func (x nat) bitLen() int {
	// This function is used in cryptographic operations. It must not leak
	// anything but the Int's sign and bit size through side-channels. Any
	// changes must be reviewed by a security expert.
	if i := len(x) - 1; i >= 0 {
		// bits.Len uses a lookup table for the low-order bits on some
		// architectures. Neutralize any input-dependent behavior by setting all
		// bits after the first one bit.
		top := uint(x[i])
		top |= top >> 1
		top |= top >> 2
		top |= top >> 4
		top |= top >> 8
		top |= top >> 16
		top |= top >> 16 >> 16 // ">> 32" doesn't compile on 32-bit architectures
		return i*_W + bits.Len(top)
	}
	return 0
}

// trailingZeroBits returns the number of consecutive least significant zero
// bits of x.
func (x nat) trailingZeroBits() uint {
	if len(x) == 0 {
		return 0
	}
	var i uint
	for x[i] == 0 {
		i++
	}
	// x[i] != 0
	return i*_W + uint(bits.TrailingZeros(uint(x[i])))
}

// isPow2 returns i, true when x == 2**i and 0, false otherwise.
func (x nat) isPow2() (uint, bool) {
	var i uint
	for x[i] == 0 {
		i++
	}
	if i == uint(len(x))-1 && x[i]&(x[i]-1) == 0 {
		return i*_W + uint(bits.TrailingZeros(uint(x[i]))), true
	}
	return 0, false
}

func same(x, y nat) bool {
	return len(x) == len(y) && len(x) > 0 && &x[0] == &y[0]
}

// z = x << s
func (z nat) shl(x nat, s uint) nat {
	if s == 0 {
		if same(z, x) {
			return z
		}
		if !alias(z, x) {
			return z.set(x)
		}
	}

	m := len(x)
	if m == 0 {
		return z[:0]
	}
	// m > 0

	n := m + int(s/_W)
	z = z.make(n + 1)
	z[n] = shlVU(z[n-m:n], x, s%_W)
	clear(z[0 : n-m])

	return z.norm()
}

// z = x >> s
func (z nat) shr(x nat, s uint) nat {
	if s == 0 {
		if same(z, x) {
			return z
		}
		if !alias(z, x) {
			return z.set(x)
		}
	}

	m := len(x)
	n := m - int(s/_W)
	if n <= 0 {
		return z[:0]
	}
	// n > 0

	z = z.make(n)
	shrVU(z, x[m-n:], s%_W)

	return z.norm()
}

func (z nat) setBit(x nat, i uint, b uint) nat {
	j := int(i / _W)
	m := Word(1) << (i % _W)
	n := len(x)
	switch b {
	case 0:
		z = z.make(n)
		copy(z, x)
		if j >= n {
			// no need to grow
			return z
		}
		z[j] &^= m
		return z.norm()
	case 1:
		if j >= n {
			z = z.make(j + 1)
			clear(z[n:])
		} else {
			z = z.make(n)
		}
		copy(z, x)
		z[j] |= m
		// no need to normalize
		return z
	}
	panic("set bit is not 0 or 1")
}

// bit returns the value of the i'th bit, with lsb == bit 0.
func (x nat) bit(i uint) uint {
	j := i / _W
	if j >= uint(len(x)) {
		return 0
	}
	// 0 <= j < len(x)
	return uint(x[j] >> (i % _W) & 1)
}

// sticky returns 1 if there's a 1 bit within the
// i least significant bits, otherwise it returns 0.
func (x nat) sticky(i uint) uint {
	j := i / _W
	if j >= uint(len(x)) {
		if len(x) == 0 {
			return 0
		}
		return 1
	}
	// 0 <= j < len(x)
	for _, x := range x[:j] {
		if x != 0 {
			return 1
		}
	}
	if x[j]<<(_W-i%_W) != 0 {
		return 1
	}
	return 0
}

func (z nat) and(x, y nat) nat {
	m := len(x)
	n := len(y)
	if m > n {
		m = n
	}
	// m <= n

	z = z.make(m)
	for i := 0; i < m; i++ {
		z[i] = x[i] & y[i]
	}

	return z.norm()
}

// trunc returns z = x mod 2ⁿ.
func (z nat) trunc(x nat, n uint) nat {
	w := (n + _W - 1) / _W
	if uint(len(x)) < w {
		return z.set(x)
	}
	z = z.make(int(w))
	copy(z, x)
	if n%_W != 0 {
		z[len(z)-1] &= 1<<(n%_W) - 1
	}
	return z.norm()
}

func (z nat) andNot(x, y nat) nat {
	m := len(x)
	n := len(y)
	if n > m {
		n = m
	}
	// m >= n

	z = z.make(m)
	for i := 0; i < n; i++ {
		z[i] = x[i] &^ y[i]
	}
	copy(z[n:m], x[n:m])

	return z.norm()
}

func (z nat) or(x, y nat) nat {
	m := len(x)
	n := len(y)
	s := x
	if m < n {
		n, m = m, n
		s = y
	}
	// m >= n

	z = z.make(m)
	for i := 0; i < n; i++ {
		z[i] = x[i] | y[i]
	}
	copy(z[n:m], s[n:m])

	return z.norm()
}

func (z nat) xor(x, y nat) nat {
	m := len(x)
	n := len(y)
	s := x
	if m < n {
		n, m = m, n
		s = y
	}
	// m >= n

	z = z.make(m)
	for i := 0; i < n; i++ {
		z[i] = x[i] ^ y[i]
	}
	copy(z[n:m], s[n:m])

	return z.norm()
}

// random creates a random integer in [0..limit), using the space in z if
// possible. n is the bit length of limit.
func (z nat) random(rand *rand.Rand, limit nat, n int) nat {
	if alias(z, limit) {
		z = nil // z is an alias for limit - cannot reuse
	}
	z = z.make(len(limit))

	bitLengthOfMSW := uint(n % _W)
	if bitLengthOfMSW == 0 {
		bitLengthOfMSW = _W
	}
	mask := Word((1 << bitLengthOfMSW) - 1)

	for {
		switch _W {
		case 32:
			for i := range z {
				z[i] = Word(rand.Uint32())
			}
		case 64:
			for i := range z {
				z[i] = Word(rand.Uint32()) | Word(rand.Uint32())<<32
			}
		default:
			panic("unknown word size")
		}
		z[len(limit)-1] &= mask
		if z.cmp(limit) < 0 {
			break
		}
	}

	return z.norm()
}

// If m != 0 (i.e., len(m) != 0), expNN sets z to x**y mod m;
// otherwise it sets z to x**y. The result is the value of z.
// The caller may pass stk == nil to request that expNN obtain and release one itself.
func (z nat) expNN(stk *stack, x, y, m nat, slow bool) nat {
	if alias(z, x) || alias(z, y) {
		// We cannot allow in-place modification of x or y.
		z = nil
	}

	// x**y mod 1 == 0
	if len(m) == 1 && m[0] == 1 {
		return z.setWord(0)
	}
	// m == 0 || m > 1

	// x**0 == 1
	if len(y) == 0 {
		return z.setWord(1)
	}
	// y > 0

	// 0**y = 0
	if len(x) == 0 {
		return z.setWord(0)
	}
	// x > 0

	// 1**y = 1
	if len(x) == 1 && x[0] == 1 {
		return z.setWord(1)
	}
	// x > 1

	// x**1 == x
	if len(y) == 1 && y[0] == 1 && len(m) == 0 {
		return z.set(x)
	}
	if stk == nil {
		stk = getStack()
		defer stk.free()
	}
	if len(y) == 1 && y[0] == 1 { // len(m) > 0
		return z.rem(stk, x, m)
	}

	// y > 1

	if len(m) != 0 {
		// We likely end up being as long as the modulus.
		z = z.make(len(m))

		// If the exponent is large, we use the Montgomery method for odd values,
		// and a 4-bit, windowed exponentiation for powers of two,
		// and a CRT-decomposed Montgomery method for the remaining values
		// (even values times non-trivial odd values, which decompose into one
		// instance of each of the first two cases).
		if len(y) > 1 && !slow {
			if m[0]&1 == 1 {
				return z.expNNMontgomery(stk, x, y, m)
			}
			if logM, ok := m.isPow2(); ok {
				return z.expNNWindowed(stk, x, y, logM)
			}
			return z.expNNMontgomeryEven(stk, x, y, m)
		}
	}

	z = z.set(x)
	v := y[len(y)-1] // v > 0 because y is normalized and y > 0
	shift := nlz(v) + 1
	v <<= shift
	var q nat

	const mask = 1 << (_W - 1)

	// We walk through the bits of the exponent one by one. Each time we
	// see a bit, we square, thus doubling the power. If the bit is a one,
	// we also multiply by x, thus adding one to the power.

	w := _W - int(shift)
	// zz and r are used to avoid allocating in mul and div as
	// otherwise the arguments would alias.
	var zz, r nat
	for j := 0; j < w; j++ {
		zz = zz.sqr(stk, z)
		zz, z = z, zz

		if v&mask != 0 {
			zz = zz.mul(stk, z, x)
			zz, z = z, zz
		}

		if len(m) != 0 {
			zz, r = zz.div(stk, r, z, m)
			zz, r, q, z = q, z, zz, r
		}

		v <<= 1
	}

	for i := len(y) - 2; i >= 0; i-- {
		v = y[i]

		for j := 0; j < _W; j++ {
			zz = zz.sqr(stk, z)
			zz, z = z, zz

			if v&mask != 0 {
				zz = zz.mul(stk, z, x)
				zz, z = z, zz
			}

			if len(m) != 0 {
				zz, r = zz.div(stk, r, z, m)
				zz, r, q, z = q, z, zz, r
			}

			v <<= 1
		}
	}

	return z.norm()
}

// expNNMontgomeryEven calculates x**y mod m where m = m1 × m2 for m1 = 2ⁿ and m2 odd.
// It uses two recursive calls to expNN for x**y mod m1 and x**y mod m2
// and then uses the Chinese Remainder Theorem to combine the results.
// The recursive call using m1 will use expNNWindowed,
// while the recursive call using m2 will use expNNMontgomery.
// For more details, see Ç. K. Koç, “Montgomery Reduction with Even Modulus”,
// IEE Proceedings: Computers and Digital Techniques, 141(5) 314-316, September 1994.
// http://www.people.vcu.edu/~jwang3/CMSC691/j34monex.pdf
func (z nat) expNNMontgomeryEven(stk *stack, x, y, m nat) nat {
	// Split m = m₁ × m₂ where m₁ = 2ⁿ
	n := m.trailingZeroBits()
	m1 := nat(nil).shl(natOne, n)
	m2 := nat(nil).shr(m, n)

	// We want z = x**y mod m.
	// z₁ = x**y mod m1 = (x**y mod m) mod m1 = z mod m1
	// z₂ = x**y mod m2 = (x**y mod m) mod m2 = z mod m2
	// (We are using the math/big convention for names here,
	// where the computation is z = x**y mod m, so its parts are z1 and z2.
	// The paper is computing x = a**e mod n; it refers to these as x2 and z1.)
	z1 := nat(nil).expNN(stk, x, y, m1, false)
	z2 := nat(nil).expNN(stk, x, y, m2, false)

	// Reconstruct z from z₁, z₂ using CRT, using algorithm from paper,
	// which uses only a single modInverse (and an easy one at that).
	//	p = (z₁ - z₂) × m₂⁻¹ (mod m₁)
	//	z = z₂ + p × m₂
	// The final addition is in range because:
	//	z = z₂ + p × m₂
	//	  ≤ z₂ + (m₁-1) × m₂
	//	  < m₂ + (m₁-1) × m₂
	//	  = m₁ × m₂
	//	  = m.
	z = z.set(z2)

	// Compute (z₁ - z₂) mod m1 [m1 == 2**n] into z1.
	z1 = z1.subMod2N(z1, z2, n)

	// Reuse z2 for p = (z₁ - z₂) [in z1] * m2⁻¹ (mod m₁ [= 2ⁿ]).
	m2inv := nat(nil).modInverse(m2, m1)
	z2 = z2.mul(stk, z1, m2inv)
	z2 = z2.trunc(z2, n)

	// Reuse z1 for p * m2.
	z = z.add(z, z1.mul(stk, z2, m2))

	return z
}

// expNNWindowed calculates x**y mod m using a fixed, 4-bit window,
// where m = 2**logM.
func (z nat) expNNWindowed(stk *stack, x, y nat, logM uint) nat {
	if len(y) <= 1 {
		panic("big: misuse of expNNWindowed")
	}
	if x[0]&1 == 0 {
		// len(y) > 1, so y  > logM.
		// x is even, so x**y is a multiple of 2**y which is a multiple of 2**logM.
		return z.setWord(0)
	}
	if logM == 1 {
		return z.setWord(1)
	}

	// zz is used to avoid allocating in mul as otherwise
	// the arguments would alias.
	defer stk.restore(stk.save())
	w := int((logM + _W - 1) / _W)
	zz := stk.nat(w)

	const n = 4
	// powers[i] contains x^i.
	var powers [1 << n]nat
	for i := range powers {
		powers[i] = stk.nat(w)
	}
	powers[0] = powers[0].set(natOne)
	powers[1] = powers[1].trunc(x, logM)
	for i := 2; i < 1<<n; i += 2 {
		p2, p, p1 := &powers[i/2], &powers[i], &powers[i+1]
		*p = p.sqr(stk, *p2)
		*p = p.trunc(*p, logM)
		*p1 = p1.mul(stk, *p, x)
		*p1 = p1.trunc(*p1, logM)
	}

	// Because phi(2**logM) = 2**(logM-1), x**(2**(logM-1)) = 1,
	// so we can compute x**(y mod 2**(logM-1)) instead of x**y.
	// That is, we can throw away all but the bottom logM-1 bits of y.
	// Instead of allocating a new y, we start reading y at the right word
	// and truncate it appropriately at the start of the loop.
	i := len(y) - 1
	mtop := int((logM - 2) / _W) // -2 because the top word of N bits is the (N-1)/W'th word.
	mmask := ^Word(0)
	if mbits := (logM - 1) & (_W - 1); mbits != 0 {
		mmask = (1 << mbits) - 1
	}
	if i > mtop {
		i = mtop
	}
	advance := false
	z = z.setWord(1)
	for ; i >= 0; i-- {
		yi := y[i]
		if i == mtop {
			yi &= mmask
		}
		for j := 0; j < _W; j += n {
			if advance {
				// Account for use of 4 bits in previous iteration.
				// Unrolled loop for significant performance
				// gain. Use go test -bench=".*" in crypto/rsa
				// to check performance before making changes.
				zz = zz.sqr(stk, z)
				zz, z = z, zz
				z = z.trunc(z, logM)

				zz = zz.sqr(stk, z)
				zz, z = z, zz
				z = z.trunc(z, logM)

				zz = zz.sqr(stk, z)
				zz, z = z, zz
				z = z.trunc(z, logM)

				zz = zz.sqr(stk, z)
				zz, z = z, zz
				z = z.trunc(z, logM)
			}

			zz = zz.mul(stk, z, powers[yi>>(_W-n)])
			zz, z = z, zz
			z = z.trunc(z, logM)

			yi <<= n
			advance = true
		}
	}

	return z.norm()
}

// expNNMontgomery calculates x**y mod m using a fixed, 4-bit window.
// Uses Montgomery representation.
func (z nat) expNNMontgomery(stk *stack, x, y, m nat) nat {
	numWords := len(m)

	// We want the lengths of x and m to be equal.
	// It is OK if x >= m as long as len(x) == len(m).
	if len(x) > numWords {
		_, x = nat(nil).div(stk, nil, x, m)
		// Note: now len(x) <= numWords, not guaranteed ==.
	}
	if len(x) < numWords {
		rr := make(nat, numWords)
		copy(rr, x)
		x = rr
	}

	// Ideally the precomputations would be performed outside, and reused
	// k0 = -m**-1 mod 2**_W. Algorithm from: Dumas, J.G. "On Newton–Raphson
	// Iteration for Multiplicative Inverses Modulo Prime Powers".
	k0 := 2 - m[0]
	t := m[0] - 1
	for i := 1; i < _W; i <<= 1 {
		t *= t
		k0 *= (t + 1)
	}
	k0 = -k0

	// RR = 2**(2*_W*len(m)) mod m
	RR := nat(nil).setWord(1)
	zz := nat(nil).shl(RR, uint(2*numWords*_W))
	_, RR = nat(nil).div(stk, RR, zz, m)
	if len(RR) < numWords {
		zz = zz.make(numWords)
		copy(zz, RR)
		RR = zz
	}
	// one = 1, with equal length to that of m
	one := make(nat, numWords)
	one[0] = 1

	const n = 4
	// powers[i] contains x^i
	var powers [1 << n]nat
	powers[0] = powers[0].montgomery(one, RR, m, k0, numWords)
	powers[1] = powers[1].montgomery(x, RR, m, k0, numWords)
	for i := 2; i < 1<<n; i++ {
		powers[i] = powers[i].montgomery(powers[i-1], powers[1], m, k0, numWords)
	}

	// initialize z = 1 (Montgomery 1)
	z = z.make(numWords)
	copy(z, powers[0])

	zz = zz.make(numWords)

	// same windowed exponent, but with Montgomery multiplications
	for i := len(y) - 1; i >= 0; i-- {
		yi := y[i]
		for j := 0; j < _W; j += n {
			if i != len(y)-1 || j != 0 {
				zz = zz.montgomery(z, z, m, k0, numWords)
				z = z.montgomery(zz, zz, m, k0, numWords)
				zz = zz.montgomery(z, z, m, k0, numWords)
				z = z.montgomery(zz, zz, m, k0, numWords)
			}
			zz = zz.montgomery(z, powers[yi>>(_W-n)], m, k0, numWords)
			z, zz = zz, z
			yi <<= n
		}
	}
	// convert to regular number
	zz = zz.montgomery(z, one, m, k0, numWords)

	// One last reduction, just in case.
	// See golang.org/issue/13907.
	if zz.cmp(m) >= 0 {
		// Common case is m has high bit set; in that case,
		// since zz is the same length as m, there can be just
		// one multiple of m to remove. Just subtract.
		// We think that the subtract should be sufficient in general,
		// so do that unconditionally, but double-check,
		// in case our beliefs are wrong.
		// The div is not expected to be reached.
		zz = zz.sub(zz, m)
		if zz.cmp(m) >= 0 {
			_, zz = nat(nil).div(stk, nil, zz, m)
		}
	}

	return zz.norm()
}

// bytes writes the value of z into buf using big-endian encoding.
// The value of z is encoded in the slice buf[i:]. If the value of z
// cannot be represented in buf, bytes panics. The number i of unused
// bytes at the beginning of buf is returned as result.
func (z nat) bytes(buf []byte) (i int) {
	// This function is used in cryptographic operations. It must not leak
	// anything but the Int's sign and bit size through side-channels. Any
	// changes must be reviewed by a security expert.
	i = len(buf)
	for _, d := range z {
		for j := 0; j < _S; j++ {
			i--
			if i >= 0 {
				buf[i] = byte(d)
			} else if byte(d) != 0 {
				panic("math/big: buffer too small to fit value")
			}
			d >>= 8
		}
	}

	if i < 0 {
		i = 0
	}
	for i < len(buf) && buf[i] == 0 {
		i++
	}

	return
}

// bigEndianWord returns the contents of buf interpreted as a big-endian encoded Word value.
func bigEndianWord(buf []byte) Word {
	if _W == 64 {
		return Word(byteorder.BEUint64(buf))
	}
	return Word(byteorder.BEUint32(buf))
}

// setBytes interprets buf as the bytes of a big-endian unsigned
// integer, sets z to that value, and returns z.
func (z nat) setBytes(buf []byte) nat {
	z = z.make((len(buf) + _S - 1) / _S)

	i := len(buf)
	for k := 0; i >= _S; k++ {
		z[k] = bigEndianWord(buf[i-_S : i])
		i -= _S
	}
	if i > 0 {
		var d Word
		for s := uint(0); i > 0; s += 8 {
			d |= Word(buf[i-1]) << s
			i--
		}
		z[len(z)-1] = d
	}

	return z.norm()
}

// sqrt sets z = ⌊√x⌋
// The caller may pass stk == nil to request that sqrt obtain and release one itself.
func (z nat) sqrt(stk *stack, x nat) nat {
	if x.cmp(natOne) <= 0 {
		return z.set(x)
	}
	if alias(z, x) {
		z = nil
	}

	if stk == nil {
		stk = getStack()
		defer stk.free()
	}

	// Start with value known to be too large and repeat "z = ⌊(z + ⌊x/z⌋)/2⌋" until it stops getting smaller.
	// See Brent and Zimmermann, Modern Computer Arithmetic, Algorithm 1.13 (SqrtInt).
	// https://members.loria.fr/PZimmermann/mca/pub226.html
	// If x is one less than a perfect square, the sequence oscillates between the correct z and z+1;
	// otherwise it converges to the correct z and stays there.
	var z1, z2 nat
	z1 = z
	z1 = z1.setUint64(1)
	z1 = z1.shl(z1, uint(x.bitLen()+1)/2) // must be ≥ √x
	for n := 0; ; n++ {
		z2, _ = z2.div(stk, nil, x, z1)
		z2 = z2.add(z2, z1)
		z2 = z2.shr(z2, 1)
		if z2.cmp(z1) >= 0 {
			// z1 is answer.
			// Figure out whether z1 or z2 is currently aliased to z by looking at loop count.
			if n&1 == 0 {
				return z1
			}
			return z.set(z1)
		}
		z1, z2 = z2, z1
	}
}

// subMod2N returns z = (x - y) mod 2ⁿ.
func (z nat) subMod2N(x, y nat, n uint) nat {
	if uint(x.bitLen()) > n {
		if alias(z, x) {
			// ok to overwrite x in place
			x = x.trunc(x, n)
		} else {
			x = nat(nil).trunc(x, n)
		}
	}
	if uint(y.bitLen()) > n {
		if alias(z, y) {
			// ok to overwrite y in place
			y = y.trunc(y, n)
		} else {
			y = nat(nil).trunc(y, n)
		}
	}
	if x.cmp(y) >= 0 {
		return z.sub(x, y)
	}
	// x - y < 0; x - y mod 2ⁿ = x - y + 2ⁿ = 2ⁿ - (y - x) = 1 + 2ⁿ-1 - (y - x) = 1 + ^(y - x).
	z = z.sub(y, x)
	for uint(len(z))*_W < n {
		z = append(z, 0)
	}
	for i := range z {
		z[i] = ^z[i]
	}
	z = z.trunc(z, n)
	return z.add(z, natOne)
}

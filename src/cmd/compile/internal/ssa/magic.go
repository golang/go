// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"math/big"
	"math/bits"
)

// So you want to compute x / c for some constant c?
// Machine division instructions are slow, so we try to
// compute this division with a multiplication + a few
// other cheap instructions instead.
// (We assume here that c != 0, +/- 1, or +/- 2^i.  Those
// cases are easy to handle in different ways).

// Technique from https://gmplib.org/~tege/divcnst-pldi94.pdf

// First consider unsigned division.
// Our strategy is to precompute 1/c then do
//   ⎣x / c⎦ = ⎣x * (1/c)⎦.
// 1/c is less than 1, so we can't compute it directly in
// integer arithmetic.  Let's instead compute 2^e/c
// for a value of e TBD (^ = exponentiation).  Then
//   ⎣x / c⎦ = ⎣x * (2^e/c) / 2^e⎦.
// Dividing by 2^e is easy.  2^e/c isn't an integer, unfortunately.
// So we must approximate it.  Let's call its approximation m.
// We'll then compute
//   ⎣x * m / 2^e⎦
// Which we want to be equal to ⎣x / c⎦ for 0 <= x < 2^n-1
// where n is the word size.
// Setting x = c gives us c * m >= 2^e.
// We'll chose m = ⎡2^e/c⎤ to satisfy that equation.
// What remains is to choose e.
// Let m = 2^e/c + delta, 0 <= delta < 1
//   ⎣x * (2^e/c + delta) / 2^e⎦
//   ⎣x / c + x * delta / 2^e⎦
// We must have x * delta / 2^e < 1/c so that this
// additional term never rounds differently than ⎣x / c⎦ does.
// Rearranging,
//   2^e > x * delta * c
// x can be at most 2^n-1 and delta can be at most 1.
// So it is sufficient to have 2^e >= 2^n*c.
// So we'll choose e = n + s, with s = ⎡log2(c)⎤.
//
// An additional complication arises because m has n+1 bits in it.
// Hardware restricts us to n bit by n bit multiplies.
// We divide into 3 cases:
//
// Case 1: m is even.
//   ⎣x / c⎦ = ⎣x * m / 2^(n+s)⎦
//   ⎣x / c⎦ = ⎣x * (m/2) / 2^(n+s-1)⎦
//   ⎣x / c⎦ = ⎣x * (m/2) / 2^n / 2^(s-1)⎦
//   ⎣x / c⎦ = ⎣⎣x * (m/2) / 2^n⎦ / 2^(s-1)⎦
//   multiply + shift
//
// Case 2: c is even.
//   ⎣x / c⎦ = ⎣(x/2) / (c/2)⎦
//   ⎣x / c⎦ = ⎣⎣x/2⎦ / (c/2)⎦
//     This is just the original problem, with x' = ⎣x/2⎦, c' = c/2, n' = n-1.
//       s' = s-1
//       m' = ⎡2^(n'+s')/c'⎤
//          = ⎡2^(n+s-1)/c⎤
//          = ⎡m/2⎤
//   ⎣x / c⎦ = ⎣x' * m' / 2^(n'+s')⎦
//   ⎣x / c⎦ = ⎣⎣x/2⎦ * ⎡m/2⎤ / 2^(n+s-2)⎦
//   ⎣x / c⎦ = ⎣⎣⎣x/2⎦ * ⎡m/2⎤ / 2^n⎦ / 2^(s-2)⎦
//   shift + multiply + shift
//
// Case 3: everything else
//   let k = m - 2^n. k fits in n bits.
//   ⎣x / c⎦ = ⎣x * m / 2^(n+s)⎦
//   ⎣x / c⎦ = ⎣x * (2^n + k) / 2^(n+s)⎦
//   ⎣x / c⎦ = ⎣(x + x * k / 2^n) / 2^s⎦
//   ⎣x / c⎦ = ⎣(x + ⎣x * k / 2^n⎦) / 2^s⎦
//   ⎣x / c⎦ = ⎣(x + ⎣x * k / 2^n⎦) / 2^s⎦
//   ⎣x / c⎦ = ⎣⎣(x + ⎣x * k / 2^n⎦) / 2⎦ / 2^(s-1)⎦
//   multiply + avg + shift
//
// These can be implemented in hardware using:
//  ⎣a * b / 2^n⎦ - aka high n bits of an n-bit by n-bit multiply.
//  ⎣(a+b) / 2⎦   - aka "average" of two n-bit numbers.
//                  (Not just a regular add & shift because the intermediate result
//                   a+b has n+1 bits in it.  Nevertheless, can be done
//                   in 2 instructions on x86.)

// umagicOK reports whether we should strength reduce a n-bit divide by c.
func umagicOK(n uint, c int64) bool {
	// Convert from ConstX auxint values to the real uint64 constant they represent.
	d := uint64(c) << (64 - n) >> (64 - n)

	// Doesn't work for 0.
	// Don't use for powers of 2.
	return d&(d-1) != 0
}

// umagicOKn reports whether we should strength reduce an unsigned n-bit divide by c.
// We can strength reduce when c != 0 and c is not a power of two.
func umagicOK8(c int8) bool   { return c&(c-1) != 0 }
func umagicOK16(c int16) bool { return c&(c-1) != 0 }
func umagicOK32(c int32) bool { return c&(c-1) != 0 }
func umagicOK64(c int64) bool { return c&(c-1) != 0 }

type umagicData struct {
	s int64  // ⎡log2(c)⎤
	m uint64 // ⎡2^(n+s)/c⎤ - 2^n
}

// umagic computes the constants needed to strength reduce unsigned n-bit divides by the constant uint64(c).
// The return values satisfy for all 0 <= x < 2^n
//
//	floor(x / uint64(c)) = x * (m + 2^n) >> (n+s)
func umagic(n uint, c int64) umagicData {
	// Convert from ConstX auxint values to the real uint64 constant they represent.
	d := uint64(c) << (64 - n) >> (64 - n)

	C := new(big.Int).SetUint64(d)
	s := C.BitLen()
	M := big.NewInt(1)
	M.Lsh(M, n+uint(s))     // 2^(n+s)
	M.Add(M, C)             // 2^(n+s)+c
	M.Sub(M, big.NewInt(1)) // 2^(n+s)+c-1
	M.Div(M, C)             // ⎡2^(n+s)/c⎤
	if M.Bit(int(n)) != 1 {
		panic("n+1st bit isn't set")
	}
	M.SetBit(M, int(n), 0)
	m := M.Uint64()
	return umagicData{s: int64(s), m: m}
}

func umagic8(c int8) umagicData   { return umagic(8, int64(c)) }
func umagic16(c int16) umagicData { return umagic(16, int64(c)) }
func umagic32(c int32) umagicData { return umagic(32, int64(c)) }
func umagic64(c int64) umagicData { return umagic(64, c) }

// umagic32PreShifted returns the pre-shifted 64-bit magic constant for unsigned 32-bit
// division by c on 64-bit targets that have a native 64x64->128-bit multiply instruction
// (amd64 MULQ, arm64 UMULH, riscv64 MULHU, etc.), enabling:
//
//	x / c = Hmul64u(ZeroExt32to64(x), umagic32PreShifted(c))
//
// Given umagic32(c) returning m and s, the constant is (2^32 + m) << (32 - s).
// Valid when umagicOK32(c) is true. Result always fits in uint64.
func umagic32PreShifted(c int32) uint64 {
	magic := umagic32(c)
	return (1<<32 + magic.m) << uint(32-magic.s)
}

// For signed division, we use a similar strategy.
// First, we enforce a positive c.
//   x / c = -(x / (-c))
// This will require an additional Neg op for c<0.
//
// If x is positive we're in a very similar state
// to the unsigned case above.  We define:
//   s = ⎡log2(c)⎤-1
//   m = ⎡2^(n+s)/c⎤
// Then
//   ⎣x / c⎦ = ⎣x * m / 2^(n+s)⎦
// If x is negative we have
//   ⎡x / c⎤ = ⎣x * m / 2^(n+s)⎦ + 1
// (TODO: derivation?)
//
// The multiply is a bit odd, as it is a signed n-bit value
// times an unsigned n-bit value.  For n smaller than the
// word size, we can extend x and m appropriately and use the
// signed multiply instruction.  For n == word size,
// we must use the signed multiply high and correct
// the result by adding x*2^n.
//
// Adding 1 if x<0 is done by subtracting x>>(n-1).

func smagicOK(n uint, c int64) bool {
	if c < 0 {
		// Doesn't work for negative c.
		return false
	}
	// Doesn't work for 0.
	// Don't use it for powers of 2.
	return c&(c-1) != 0
}

// smagicOKn reports whether we should strength reduce a signed n-bit divide by c.
func smagicOK8(c int8) bool   { return smagicOK(8, int64(c)) }
func smagicOK16(c int16) bool { return smagicOK(16, int64(c)) }
func smagicOK32(c int32) bool { return smagicOK(32, int64(c)) }
func smagicOK64(c int64) bool { return smagicOK(64, c) }

type smagicData struct {
	s int64  // ⎡log2(c)⎤-1
	m uint64 // ⎡2^(n+s)/c⎤
}

// smagic computes the constants needed to strength reduce signed n-bit divides by the constant c.
// Must have c>0.
// The return values satisfy for all -2^(n-1) <= x < 2^(n-1)
//
//	trunc(x / c) = x * m >> (n+s) + (x < 0 ? 1 : 0)
func smagic(n uint, c int64) smagicData {
	C := new(big.Int).SetInt64(c)
	s := C.BitLen() - 1
	M := big.NewInt(1)
	M.Lsh(M, n+uint(s))     // 2^(n+s)
	M.Add(M, C)             // 2^(n+s)+c
	M.Sub(M, big.NewInt(1)) // 2^(n+s)+c-1
	M.Div(M, C)             // ⎡2^(n+s)/c⎤
	if M.Bit(int(n)) != 0 {
		panic("n+1st bit is set")
	}
	if M.Bit(int(n-1)) == 0 {
		panic("nth bit is not set")
	}
	m := M.Uint64()
	return smagicData{s: int64(s), m: m}
}

func smagic8(c int8) smagicData   { return smagic(8, int64(c)) }
func smagic16(c int16) smagicData { return smagic(16, int64(c)) }
func smagic32(c int32) smagicData { return smagic(32, int64(c)) }
func smagic64(c int64) smagicData { return smagic(64, c) }

// Divisibility x%c == 0 can be checked more efficiently than directly computing
// the modulus x%c and comparing against 0.
//
// The same "Division by invariant integers using multiplication" paper
// by Granlund and Montgomery referenced above briefly mentions this method
// and it is further elaborated in "Hacker's Delight" by Warren Section 10-17
//
// The first thing to note is that for odd integers, exact division can be computed
// by using the modular inverse with respect to the word size 2^n.
//
// Given c, compute m such that (c * m) mod 2^n == 1
// Then if c divides x (x%c ==0), the quotient is given by q = x/c == x*m mod 2^n
//
// x can range from 0, c, 2c, 3c, ... ⎣(2^n - 1)/c⎦ * c the maximum multiple
// Thus, x*m mod 2^n is 0, 1, 2, 3, ... ⎣(2^n - 1)/c⎦
// i.e. the quotient takes all values from zero up to max = ⎣(2^n - 1)/c⎦
//
// If x is not divisible by c, then x*m mod 2^n must take some larger value than max.
//
// This gives x*m mod 2^n <= ⎣(2^n - 1)/c⎦ as a test for divisibility
// involving one multiplication and compare.
//
// To extend this to even integers, consider c = d0 * 2^k where d0 is odd.
// We can test whether x is divisible by both d0 and 2^k.
// For d0, the test is the same as above.  Let m be such that m*d0 mod 2^n == 1
// Then x*m mod 2^n <= ⎣(2^n - 1)/d0⎦ is the first test.
// The test for divisibility by 2^k is a check for k trailing zeroes.
// Note that since d0 is odd, m is odd and thus x*m will have the same number of
// trailing zeroes as x.  So the two tests are,
//
// x*m mod 2^n <= ⎣(2^n - 1)/d0⎦
// and x*m ends in k zero bits
//
// These can be combined into a single comparison by the following
// (theorem ZRU in Hacker's Delight) for unsigned integers.
//
// x <= a and x ends in k zero bits if and only if RotRight(x ,k) <= ⎣a/(2^k)⎦
// Where RotRight(x ,k) is right rotation of x by k bits.
//
// To prove the first direction, x <= a -> ⎣x/(2^k)⎦ <= ⎣a/(2^k)⎦
// But since x ends in k zeroes all the rotated bits would be zero too.
// So RotRight(x, k) == ⎣x/(2^k)⎦ <= ⎣a/(2^k)⎦
//
// If x does not end in k zero bits, then RotRight(x, k)
// has some non-zero bits in the k highest bits.
// ⎣x/(2^k)⎦ has all zeroes in the k highest bits,
// so RotRight(x, k) > ⎣x/(2^k)⎦
//
// Finally, if x > a and has k trailing zero bits, then RotRight(x, k) == ⎣x/(2^k)⎦
// and ⎣x/(2^k)⎦ must be greater than ⎣a/(2^k)⎦, that is the top n-k bits of x must
// be greater than the top n-k bits of a because the rest of x bits are zero.
//
// So the two conditions about can be replaced with the single test
//
// RotRight(x*m mod 2^n, k) <= ⎣(2^n - 1)/c⎦
//
// Where d0*2^k was replaced by c on the right hand side.

// udivisibleOK reports whether we should strength reduce an unsigned n-bit divisibility check by c.
func udivisibleOK(n uint, c int64) bool {
	// Convert from ConstX auxint values to the real uint64 constant they represent.
	d := uint64(c) << (64 - n) >> (64 - n)

	// Doesn't work for 0.
	// Don't use for powers of 2.
	return d&(d-1) != 0
}

func udivisibleOK8(c int8) bool   { return udivisibleOK(8, int64(c)) }
func udivisibleOK16(c int16) bool { return udivisibleOK(16, int64(c)) }
func udivisibleOK32(c int32) bool { return udivisibleOK(32, int64(c)) }
func udivisibleOK64(c int64) bool { return udivisibleOK(64, c) }

type udivisibleData struct {
	k   int64  // trailingZeros(c)
	m   uint64 // m * (c>>k) mod 2^n == 1 multiplicative inverse of odd portion modulo 2^n
	max uint64 // ⎣(2^n - 1)/ c⎦ max value to for divisibility
}

func udivisible(n uint, c int64) udivisibleData {
	// Convert from ConstX auxint values to the real uint64 constant they represent.
	d := uint64(c) << (64 - n) >> (64 - n)

	k := bits.TrailingZeros64(d)
	d0 := d >> uint(k) // the odd portion of the divisor

	mask := ^uint64(0) >> (64 - n)

	// Calculate the multiplicative inverse via Newton's method.
	// Quadratic convergence doubles the number of correct bits per iteration.
	m := d0            // initial guess correct to 3-bits d0*d0 mod 8 == 1
	m = m * (2 - m*d0) // 6-bits
	m = m * (2 - m*d0) // 12-bits
	m = m * (2 - m*d0) // 24-bits
	m = m * (2 - m*d0) // 48-bits
	m = m * (2 - m*d0) // 96-bits >= 64-bits
	m = m & mask

	max := mask / d

	return udivisibleData{
		k:   int64(k),
		m:   m,
		max: max,
	}
}

func udivisible8(c int8) udivisibleData   { return udivisible(8, int64(c)) }
func udivisible16(c int16) udivisibleData { return udivisible(16, int64(c)) }
func udivisible32(c int32) udivisibleData { return udivisible(32, int64(c)) }
func udivisible64(c int64) udivisibleData { return udivisible(64, c) }

// For signed integers, a similar method follows.
//
// Given c > 1 and odd, compute m such that (c * m) mod 2^n == 1
// Then if c divides x (x%c ==0), the quotient is given by q = x/c == x*m mod 2^n
//
// x can range from ⎡-2^(n-1)/c⎤ * c, ... -c, 0, c, ...  ⎣(2^(n-1) - 1)/c⎦ * c
// Thus, x*m mod 2^n is ⎡-2^(n-1)/c⎤, ... -2, -1, 0, 1, 2, ... ⎣(2^(n-1) - 1)/c⎦
//
// So, x is a multiple of c if and only if:
// ⎡-2^(n-1)/c⎤ <= x*m mod 2^n <= ⎣(2^(n-1) - 1)/c⎦
//
// Since c > 1 and odd, this can be simplified by
// ⎡-2^(n-1)/c⎤ == ⎡(-2^(n-1) + 1)/c⎤ == -⎣(2^(n-1) - 1)/c⎦
//
// -⎣(2^(n-1) - 1)/c⎦ <= x*m mod 2^n <= ⎣(2^(n-1) - 1)/c⎦
//
// To extend this to even integers, consider c = d0 * 2^k where d0 is odd.
// We can test whether x is divisible by both d0 and 2^k.
//
// Let m be such that (d0 * m) mod 2^n == 1.
// Let q = x*m mod 2^n. Then c divides x if:
//
// -⎣(2^(n-1) - 1)/d0⎦ <= q <= ⎣(2^(n-1) - 1)/d0⎦ and q ends in at least k 0-bits
//
// To transform this to a single comparison, we use the following theorem (ZRS in Hacker's Delight).
//
// For a >= 0 the following conditions are equivalent:
// 1) -a <= x <= a and x ends in at least k 0-bits
// 2) RotRight(x+a', k) <= ⎣2a'/2^k⎦
//
// Where a' = a & -2^k (a with its right k bits set to zero)
//
// To see that 1 & 2 are equivalent, note that -a <= x <= a is equivalent to
// -a' <= x <= a' if and only if x ends in at least k 0-bits.  Adding -a' to each side gives,
// 0 <= x + a' <= 2a' and x + a' ends in at least k 0-bits if and only if x does since a' has
// k 0-bits by definition.  We can use theorem ZRU above with x -> x + a' and a -> 2a' giving 1) == 2).
//
// Let m be such that (d0 * m) mod 2^n == 1.
// Let q = x*m mod 2^n.
// Let a' = ⎣(2^(n-1) - 1)/d0⎦ & -2^k
//
// Then the divisibility test is:
//
// RotRight(q+a', k) <= ⎣2a'/2^k⎦
//
// Note that the calculation is performed using unsigned integers.
// Since a' can have n-1 bits, 2a' may have n bits and there is no risk of overflow.

// sdivisibleOK reports whether we should strength reduce a signed n-bit divisibility check by c.
func sdivisibleOK(n uint, c int64) bool {
	if c < 0 {
		// Doesn't work for negative c.
		return false
	}
	// Doesn't work for 0.
	// Don't use it for powers of 2.
	return c&(c-1) != 0
}

func sdivisibleOK8(c int8) bool   { return sdivisibleOK(8, int64(c)) }
func sdivisibleOK16(c int16) bool { return sdivisibleOK(16, int64(c)) }
func sdivisibleOK32(c int32) bool { return sdivisibleOK(32, int64(c)) }
func sdivisibleOK64(c int64) bool { return sdivisibleOK(64, c) }

type sdivisibleData struct {
	k   int64  // trailingZeros(c)
	m   uint64 // m * (c>>k) mod 2^n == 1 multiplicative inverse of odd portion modulo 2^n
	a   uint64 // ⎣(2^(n-1) - 1)/ (c>>k)⎦ & -(1<<k) additive constant
	max uint64 // ⎣(2 a) / (1<<k)⎦ max value to for divisibility
}

func sdivisible(n uint, c int64) sdivisibleData {
	d := uint64(c)
	k := bits.TrailingZeros64(d)
	d0 := d >> uint(k) // the odd portion of the divisor

	mask := ^uint64(0) >> (64 - n)

	// Calculate the multiplicative inverse via Newton's method.
	// Quadratic convergence doubles the number of correct bits per iteration.
	m := d0            // initial guess correct to 3-bits d0*d0 mod 8 == 1
	m = m * (2 - m*d0) // 6-bits
	m = m * (2 - m*d0) // 12-bits
	m = m * (2 - m*d0) // 24-bits
	m = m * (2 - m*d0) // 48-bits
	m = m * (2 - m*d0) // 96-bits >= 64-bits
	m = m & mask

	a := ((mask >> 1) / d0) & -(1 << uint(k))
	max := (2 * a) >> uint(k)

	return sdivisibleData{
		k:   int64(k),
		m:   m,
		a:   a,
		max: max,
	}
}

func sdivisible8(c int8) sdivisibleData   { return sdivisible(8, int64(c)) }
func sdivisible16(c int16) sdivisibleData { return sdivisible(16, int64(c)) }
func sdivisible32(c int32) sdivisibleData { return sdivisible(32, int64(c)) }
func sdivisible64(c int64) sdivisibleData { return sdivisible(64, c) }

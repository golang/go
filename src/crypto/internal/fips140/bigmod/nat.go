// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bigmod

import (
	_ "crypto/internal/fips140/check"
	"crypto/internal/fips140deps/byteorder"
	"errors"
	"math/bits"
)

const (
	// _W is the size in bits of our limbs.
	_W = bits.UintSize
	// _S is the size in bytes of our limbs.
	_S = _W / 8
)

// choice represents a constant-time boolean. The value of choice is always
// either 1 or 0. We use an int instead of bool in order to make decisions in
// constant time by turning it into a mask.
type choice uint

func not(c choice) choice { return 1 ^ c }

const yes = choice(1)
const no = choice(0)

// ctMask is all 1s if on is yes, and all 0s otherwise.
func ctMask(on choice) uint { return -uint(on) }

// ctEq returns 1 if x == y, and 0 otherwise. The execution time of this
// function does not depend on its inputs.
func ctEq(x, y uint) choice {
	// If x != y, then either x - y or y - x will generate a carry.
	_, c1 := bits.Sub(x, y, 0)
	_, c2 := bits.Sub(y, x, 0)
	return not(choice(c1 | c2))
}

// Nat represents an arbitrary natural number
//
// Each Nat has an announced length, which is the number of limbs it has stored.
// Operations on this number are allowed to leak this length, but will not leak
// any information about the values contained in those limbs.
type Nat struct {
	// limbs is little-endian in base 2^W with W = bits.UintSize.
	limbs []uint
}

// preallocTarget is the size in bits of the numbers used to implement the most
// common and most performant RSA key size. It's also enough to cover some of
// the operations of key sizes up to 4096.
const preallocTarget = 2048
const preallocLimbs = (preallocTarget + _W - 1) / _W

// NewNat returns a new nat with a size of zero, just like new(Nat), but with
// the preallocated capacity to hold a number of up to preallocTarget bits.
// NewNat inlines, so the allocation can live on the stack.
func NewNat() *Nat {
	limbs := make([]uint, 0, preallocLimbs)
	return &Nat{limbs}
}

// expand expands x to n limbs, leaving its value unchanged.
func (x *Nat) expand(n int) *Nat {
	if len(x.limbs) > n {
		panic("bigmod: internal error: shrinking nat")
	}
	if cap(x.limbs) < n {
		newLimbs := make([]uint, n)
		copy(newLimbs, x.limbs)
		x.limbs = newLimbs
		return x
	}
	extraLimbs := x.limbs[len(x.limbs):n]
	clear(extraLimbs)
	x.limbs = x.limbs[:n]
	return x
}

// reset returns a zero nat of n limbs, reusing x's storage if n <= cap(x.limbs).
func (x *Nat) reset(n int) *Nat {
	if cap(x.limbs) < n {
		x.limbs = make([]uint, n)
		return x
	}
	clear(x.limbs)
	x.limbs = x.limbs[:n]
	return x
}

// resetToBytes assigns x = b, where b is a slice of big-endian bytes, resizing
// n to the appropriate size.
//
// The announced length of x is set based on the actual bit size of the input,
// ignoring leading zeroes.
func (x *Nat) resetToBytes(b []byte) *Nat {
	x.reset((len(b) + _S - 1) / _S)
	if err := x.setBytes(b); err != nil {
		panic("bigmod: internal error: bad arithmetic")
	}
	// Trim most significant (trailing in little-endian) zero limbs.
	// We assume comparison with zero (but not the branch) is constant time.
	for i := len(x.limbs) - 1; i >= 0; i-- {
		if x.limbs[i] != 0 {
			break
		}
		x.limbs = x.limbs[:i]
	}
	return x
}

// set assigns x = y, optionally resizing x to the appropriate size.
func (x *Nat) set(y *Nat) *Nat {
	x.reset(len(y.limbs))
	copy(x.limbs, y.limbs)
	return x
}

// Bytes returns x as a zero-extended big-endian byte slice. The size of the
// slice will match the size of m.
//
// x must have the same size as m and it must be reduced modulo m.
func (x *Nat) Bytes(m *Modulus) []byte {
	i := m.Size()
	bytes := make([]byte, i)
	for _, limb := range x.limbs {
		for j := 0; j < _S; j++ {
			i--
			if i < 0 {
				if limb == 0 {
					break
				}
				panic("bigmod: modulus is smaller than nat")
			}
			bytes[i] = byte(limb)
			limb >>= 8
		}
	}
	return bytes
}

// SetBytes assigns x = b, where b is a slice of big-endian bytes.
// SetBytes returns an error if b >= m.
//
// The output will be resized to the size of m and overwritten.
func (x *Nat) SetBytes(b []byte, m *Modulus) (*Nat, error) {
	x.resetFor(m)
	if err := x.setBytes(b); err != nil {
		return nil, err
	}
	if x.cmpGeq(m.nat) == yes {
		return nil, errors.New("input overflows the modulus")
	}
	return x, nil
}

// SetOverflowingBytes assigns x = b, where b is a slice of big-endian bytes.
// SetOverflowingBytes returns an error if b has a longer bit length than m, but
// reduces overflowing values up to 2^⌈log2(m)⌉ - 1.
//
// The output will be resized to the size of m and overwritten.
func (x *Nat) SetOverflowingBytes(b []byte, m *Modulus) (*Nat, error) {
	x.resetFor(m)
	if err := x.setBytes(b); err != nil {
		return nil, err
	}
	leading := _W - bitLen(x.limbs[len(x.limbs)-1])
	if leading < m.leading {
		return nil, errors.New("input overflows the modulus size")
	}
	x.maybeSubtractModulus(no, m)
	return x, nil
}

// bigEndianUint returns the contents of buf interpreted as a
// big-endian encoded uint value.
func bigEndianUint(buf []byte) uint {
	if _W == 64 {
		return uint(byteorder.BEUint64(buf))
	}
	return uint(byteorder.BEUint32(buf))
}

func (x *Nat) setBytes(b []byte) error {
	i, k := len(b), 0
	for k < len(x.limbs) && i >= _S {
		x.limbs[k] = bigEndianUint(b[i-_S : i])
		i -= _S
		k++
	}
	for s := 0; s < _W && k < len(x.limbs) && i > 0; s += 8 {
		x.limbs[k] |= uint(b[i-1]) << s
		i--
	}
	if i > 0 {
		return errors.New("input overflows the modulus size")
	}
	return nil
}

// SetUint assigns x = y, and returns an error if y >= m.
//
// The output will be resized to the size of m and overwritten.
func (x *Nat) SetUint(y uint, m *Modulus) (*Nat, error) {
	x.resetFor(m)
	// Modulus is never zero, so always at least one limb.
	x.limbs[0] = y
	if x.cmpGeq(m.nat) == yes {
		return nil, errors.New("input overflows the modulus")
	}
	return x, nil
}

// Equal returns 1 if x == y, and 0 otherwise.
//
// Both operands must have the same announced length.
func (x *Nat) Equal(y *Nat) choice {
	// Eliminate bounds checks in the loop.
	size := len(x.limbs)
	xLimbs := x.limbs[:size]
	yLimbs := y.limbs[:size]

	equal := yes
	for i := 0; i < size; i++ {
		equal &= ctEq(xLimbs[i], yLimbs[i])
	}
	return equal
}

// IsZero returns 1 if x == 0, and 0 otherwise.
func (x *Nat) IsZero() choice {
	// Eliminate bounds checks in the loop.
	size := len(x.limbs)
	xLimbs := x.limbs[:size]

	zero := yes
	for i := 0; i < size; i++ {
		zero &= ctEq(xLimbs[i], 0)
	}
	return zero
}

// IsOne returns 1 if x == 1, and 0 otherwise.
func (x *Nat) IsOne() choice {
	// Eliminate bounds checks in the loop.
	size := len(x.limbs)
	xLimbs := x.limbs[:size]

	if len(xLimbs) == 0 {
		return no
	}

	one := ctEq(xLimbs[0], 1)
	for i := 1; i < size; i++ {
		one &= ctEq(xLimbs[i], 0)
	}
	return one
}

// IsMinusOne returns 1 if x == -1 mod m, and 0 otherwise.
//
// The length of x must be the same as the modulus. x must already be reduced
// modulo m.
func (x *Nat) IsMinusOne(m *Modulus) choice {
	minusOne := m.Nat()
	minusOne.SubOne(m)
	return x.Equal(minusOne)
}

// IsOdd returns 1 if x is odd, and 0 otherwise.
func (x *Nat) IsOdd() choice {
	if len(x.limbs) == 0 {
		return no
	}
	return choice(x.limbs[0] & 1)
}

// TrailingZeroBitsVarTime returns the number of trailing zero bits in x.
func (x *Nat) TrailingZeroBitsVarTime() uint {
	var t uint
	limbs := x.limbs
	for _, l := range limbs {
		if l == 0 {
			t += _W
			continue
		}
		t += uint(bits.TrailingZeros(l))
		break
	}
	return t
}

// cmpGeq returns 1 if x >= y, and 0 otherwise.
//
// Both operands must have the same announced length.
func (x *Nat) cmpGeq(y *Nat) choice {
	// Eliminate bounds checks in the loop.
	size := len(x.limbs)
	xLimbs := x.limbs[:size]
	yLimbs := y.limbs[:size]

	var c uint
	for i := 0; i < size; i++ {
		_, c = bits.Sub(xLimbs[i], yLimbs[i], c)
	}
	// If there was a carry, then subtracting y underflowed, so
	// x is not greater than or equal to y.
	return not(choice(c))
}

// assign sets x <- y if on == 1, and does nothing otherwise.
//
// Both operands must have the same announced length.
func (x *Nat) assign(on choice, y *Nat) *Nat {
	// Eliminate bounds checks in the loop.
	size := len(x.limbs)
	xLimbs := x.limbs[:size]
	yLimbs := y.limbs[:size]

	mask := ctMask(on)
	for i := 0; i < size; i++ {
		xLimbs[i] ^= mask & (xLimbs[i] ^ yLimbs[i])
	}
	return x
}

// add computes x += y and returns the carry.
//
// Both operands must have the same announced length.
func (x *Nat) add(y *Nat) (c uint) {
	// Eliminate bounds checks in the loop.
	size := len(x.limbs)
	xLimbs := x.limbs[:size]
	yLimbs := y.limbs[:size]

	for i := 0; i < size; i++ {
		xLimbs[i], c = bits.Add(xLimbs[i], yLimbs[i], c)
	}
	return
}

// sub computes x -= y. It returns the borrow of the subtraction.
//
// Both operands must have the same announced length.
func (x *Nat) sub(y *Nat) (c uint) {
	// Eliminate bounds checks in the loop.
	size := len(x.limbs)
	xLimbs := x.limbs[:size]
	yLimbs := y.limbs[:size]

	for i := 0; i < size; i++ {
		xLimbs[i], c = bits.Sub(xLimbs[i], yLimbs[i], c)
	}
	return
}

// ShiftRightVarTime sets x = x >> n.
//
// The announced length of x is unchanged.
func (x *Nat) ShiftRightVarTime(n uint) *Nat {
	// Eliminate bounds checks in the loop.
	size := len(x.limbs)
	xLimbs := x.limbs[:size]

	shift := int(n % _W)
	shiftLimbs := int(n / _W)

	var shiftedLimbs []uint
	if shiftLimbs < size {
		shiftedLimbs = xLimbs[shiftLimbs:]
	}

	for i := range xLimbs {
		if i >= len(shiftedLimbs) {
			xLimbs[i] = 0
			continue
		}

		xLimbs[i] = shiftedLimbs[i] >> shift
		if i+1 < len(shiftedLimbs) {
			xLimbs[i] |= shiftedLimbs[i+1] << (_W - shift)
		}
	}

	return x
}

// Modulus is used for modular arithmetic, precomputing relevant constants.
//
// A Modulus can leak the exact number of bits needed to store its value
// and is stored without padding. Its actual value is still kept secret.
type Modulus struct {
	// The underlying natural number for this modulus.
	//
	// This will be stored without any padding, and shouldn't alias with any
	// other natural number being used.
	nat     *Nat
	leading int // number of leading zeros in the modulus

	// If m is even, the following fields are not set.
	odd   bool
	m0inv uint // -nat.limbs[0]⁻¹ mod _W
	rr    *Nat // R*R for montgomeryRepresentation
}

// rr returns R*R with R = 2^(_W * n) and n = len(m.nat.limbs).
func rr(m *Modulus) *Nat {
	rr := NewNat().ExpandFor(m)
	n := uint(len(rr.limbs))
	mLen := uint(m.BitLen())
	logR := _W * n

	// We start by computing R = 2^(_W * n) mod m. We can get pretty close, to
	// 2^⌊log₂m⌋, by setting the highest bit we can without having to reduce.
	rr.limbs[n-1] = 1 << ((mLen - 1) % _W)
	// Then we double until we reach 2^(_W * n).
	for i := mLen - 1; i < logR; i++ {
		rr.Add(rr, m)
	}

	// Next we need to get from R to 2^(_W * n) R mod m (aka from one to R in
	// the Montgomery domain, meaning we can use Montgomery multiplication now).
	// We could do that by doubling _W * n times, or with a square-and-double
	// chain log2(_W * n) long. Turns out the fastest thing is to start out with
	// doublings, and switch to square-and-double once the exponent is large
	// enough to justify the cost of the multiplications.

	// The threshold is selected experimentally as a linear function of n.
	threshold := n / 4

	// We calculate how many of the most-significant bits of the exponent we can
	// compute before crossing the threshold, and we do it with doublings.
	i := bits.UintSize
	for logR>>i <= threshold {
		i--
	}
	for k := uint(0); k < logR>>i; k++ {
		rr.Add(rr, m)
	}

	// Then we process the remaining bits of the exponent with a
	// square-and-double chain.
	for i > 0 {
		rr.montgomeryMul(rr, rr, m)
		i--
		if logR>>i&1 != 0 {
			rr.Add(rr, m)
		}
	}

	return rr
}

// minusInverseModW computes -x⁻¹ mod _W with x odd.
//
// This operation is used to precompute a constant involved in Montgomery
// multiplication.
func minusInverseModW(x uint) uint {
	// Every iteration of this loop doubles the least-significant bits of
	// correct inverse in y. The first three bits are already correct (1⁻¹ = 1,
	// 3⁻¹ = 3, 5⁻¹ = 5, and 7⁻¹ = 7 mod 8), so doubling five times is enough
	// for 64 bits (and wastes only one iteration for 32 bits).
	//
	// See https://crypto.stackexchange.com/a/47496.
	y := x
	for i := 0; i < 5; i++ {
		y = y * (2 - x*y)
	}
	return -y
}

// NewModulus creates a new Modulus from a slice of big-endian bytes. The
// modulus must be greater than one.
//
// The number of significant bits and whether the modulus is even is leaked
// through timing side-channels.
func NewModulus(b []byte) (*Modulus, error) {
	m := &Modulus{}
	m.nat = NewNat().resetToBytes(b)
	if m.nat.IsZero() == yes || m.nat.IsOne() == yes {
		return nil, errors.New("modulus must be > 1")
	}
	m.leading = _W - bitLen(m.nat.limbs[len(m.nat.limbs)-1])
	if m.nat.IsOdd() == 1 {
		m.odd = true
		m.m0inv = minusInverseModW(m.nat.limbs[0])
		m.rr = rr(m)
	}
	return m, nil
}

// bitLen is a version of bits.Len that only leaks the bit length of n, but not
// its value. bits.Len and bits.LeadingZeros use a lookup table for the
// low-order bits on some architectures.
func bitLen(n uint) int {
	var len int
	// We assume, here and elsewhere, that comparison to zero is constant time
	// with respect to different non-zero values.
	for n != 0 {
		len++
		n >>= 1
	}
	return len
}

// Size returns the size of m in bytes.
func (m *Modulus) Size() int {
	return (m.BitLen() + 7) / 8
}

// BitLen returns the size of m in bits.
func (m *Modulus) BitLen() int {
	return len(m.nat.limbs)*_W - int(m.leading)
}

// Nat returns m as a Nat.
func (m *Modulus) Nat() *Nat {
	// Make a copy so that the caller can't modify m.nat or alias it with
	// another Nat in a modulus operation.
	n := NewNat()
	n.set(m.nat)
	return n
}

// shiftIn calculates x = x << _W + y mod m.
//
// This assumes that x is already reduced mod m.
func (x *Nat) shiftIn(y uint, m *Modulus) *Nat {
	d := NewNat().resetFor(m)

	// Eliminate bounds checks in the loop.
	size := len(m.nat.limbs)
	xLimbs := x.limbs[:size]
	dLimbs := d.limbs[:size]
	mLimbs := m.nat.limbs[:size]

	// Each iteration of this loop computes x = 2x + b mod m, where b is a bit
	// from y. Effectively, it left-shifts x and adds y one bit at a time,
	// reducing it every time.
	//
	// To do the reduction, each iteration computes both 2x + b and 2x + b - m.
	// The next iteration (and finally the return line) will use either result
	// based on whether 2x + b overflows m.
	needSubtraction := no
	for i := _W - 1; i >= 0; i-- {
		carry := (y >> i) & 1
		var borrow uint
		mask := ctMask(needSubtraction)
		for i := 0; i < size; i++ {
			l := xLimbs[i] ^ (mask & (xLimbs[i] ^ dLimbs[i]))
			xLimbs[i], carry = bits.Add(l, l, carry)
			dLimbs[i], borrow = bits.Sub(xLimbs[i], mLimbs[i], borrow)
		}
		// Like in maybeSubtractModulus, we need the subtraction if either it
		// didn't underflow (meaning 2x + b > m) or if computing 2x + b
		// overflowed (meaning 2x + b > 2^_W*n > m).
		needSubtraction = not(choice(borrow)) | choice(carry)
	}
	return x.assign(needSubtraction, d)
}

// Mod calculates out = x mod m.
//
// This works regardless how large the value of x is.
//
// The output will be resized to the size of m and overwritten.
func (out *Nat) Mod(x *Nat, m *Modulus) *Nat {
	out.resetFor(m)
	// Working our way from the most significant to the least significant limb,
	// we can insert each limb at the least significant position, shifting all
	// previous limbs left by _W. This way each limb will get shifted by the
	// correct number of bits. We can insert at least N - 1 limbs without
	// overflowing m. After that, we need to reduce every time we shift.
	i := len(x.limbs) - 1
	// For the first N - 1 limbs we can skip the actual shifting and position
	// them at the shifted position, which starts at min(N - 2, i).
	start := len(m.nat.limbs) - 2
	if i < start {
		start = i
	}
	for j := start; j >= 0; j-- {
		out.limbs[j] = x.limbs[i]
		i--
	}
	// We shift in the remaining limbs, reducing modulo m each time.
	for i >= 0 {
		out.shiftIn(x.limbs[i], m)
		i--
	}
	return out
}

// ExpandFor ensures x has the right size to work with operations modulo m.
//
// The announced size of x must be smaller than or equal to that of m.
func (x *Nat) ExpandFor(m *Modulus) *Nat {
	return x.expand(len(m.nat.limbs))
}

// resetFor ensures out has the right size to work with operations modulo m.
//
// out is zeroed and may start at any size.
func (out *Nat) resetFor(m *Modulus) *Nat {
	return out.reset(len(m.nat.limbs))
}

// maybeSubtractModulus computes x -= m if and only if x >= m or if "always" is yes.
//
// It can be used to reduce modulo m a value up to 2m - 1, which is a common
// range for results computed by higher level operations.
//
// always is usually a carry that indicates that the operation that produced x
// overflowed its size, meaning abstractly x > 2^_W*n > m even if x < m.
//
// x and m operands must have the same announced length.
func (x *Nat) maybeSubtractModulus(always choice, m *Modulus) {
	t := NewNat().set(x)
	underflow := t.sub(m.nat)
	// We keep the result if x - m didn't underflow (meaning x >= m)
	// or if always was set.
	keep := not(choice(underflow)) | choice(always)
	x.assign(keep, t)
}

// Sub computes x = x - y mod m.
//
// The length of both operands must be the same as the modulus. Both operands
// must already be reduced modulo m.
func (x *Nat) Sub(y *Nat, m *Modulus) *Nat {
	underflow := x.sub(y)
	// If the subtraction underflowed, add m.
	t := NewNat().set(x)
	t.add(m.nat)
	x.assign(choice(underflow), t)
	return x
}

// SubOne computes x = x - 1 mod m.
//
// The length of x must be the same as the modulus. x must already be reduced
// modulo m.
func (x *Nat) SubOne(m *Modulus) *Nat {
	one := NewNat().ExpandFor(m)
	one.limbs[0] = 1
	return x.Sub(one, m)
}

// Add computes x = x + y mod m.
//
// The length of both operands must be the same as the modulus. Both operands
// must already be reduced modulo m.
func (x *Nat) Add(y *Nat, m *Modulus) *Nat {
	overflow := x.add(y)
	x.maybeSubtractModulus(choice(overflow), m)
	return x
}

// montgomeryRepresentation calculates x = x * R mod m, with R = 2^(_W * n) and
// n = len(m.nat.limbs).
//
// Faster Montgomery multiplication replaces standard modular multiplication for
// numbers in this representation.
//
// This assumes that x is already reduced mod m.
func (x *Nat) montgomeryRepresentation(m *Modulus) *Nat {
	// A Montgomery multiplication (which computes a * b / R) by R * R works out
	// to a multiplication by R, which takes the value out of the Montgomery domain.
	return x.montgomeryMul(x, m.rr, m)
}

// montgomeryReduction calculates x = x / R mod m, with R = 2^(_W * n) and
// n = len(m.nat.limbs).
//
// This assumes that x is already reduced mod m.
func (x *Nat) montgomeryReduction(m *Modulus) *Nat {
	// By Montgomery multiplying with 1 not in Montgomery representation, we
	// convert out back from Montgomery representation, because it works out to
	// dividing by R.
	one := NewNat().ExpandFor(m)
	one.limbs[0] = 1
	return x.montgomeryMul(x, one, m)
}

// montgomeryMul calculates x = a * b / R mod m, with R = 2^(_W * n) and
// n = len(m.nat.limbs), also known as a Montgomery multiplication.
//
// All inputs should be the same length and already reduced modulo m.
// x will be resized to the size of m and overwritten.
func (x *Nat) montgomeryMul(a *Nat, b *Nat, m *Modulus) *Nat {
	n := len(m.nat.limbs)
	mLimbs := m.nat.limbs[:n]
	aLimbs := a.limbs[:n]
	bLimbs := b.limbs[:n]

	switch n {
	default:
		// Attempt to use a stack-allocated backing array.
		T := make([]uint, 0, preallocLimbs*2)
		if cap(T) < n*2 {
			T = make([]uint, 0, n*2)
		}
		T = T[:n*2]

		// This loop implements Word-by-Word Montgomery Multiplication, as
		// described in Algorithm 4 (Fig. 3) of "Efficient Software
		// Implementations of Modular Exponentiation" by Shay Gueron
		// [https://eprint.iacr.org/2011/239.pdf].
		var c uint
		for i := 0; i < n; i++ {
			_ = T[n+i] // bounds check elimination hint

			// Step 1 (T = a × b) is computed as a large pen-and-paper column
			// multiplication of two numbers with n base-2^_W digits. If we just
			// wanted to produce 2n-wide T, we would do
			//
			//   for i := 0; i < n; i++ {
			//       d := bLimbs[i]
			//       T[n+i] = addMulVVW(T[i:n+i], aLimbs, d)
			//   }
			//
			// where d is a digit of the multiplier, T[i:n+i] is the shifted
			// position of the product of that digit, and T[n+i] is the final carry.
			// Note that T[i] isn't modified after processing the i-th digit.
			//
			// Instead of running two loops, one for Step 1 and one for Steps 2–6,
			// the result of Step 1 is computed during the next loop. This is
			// possible because each iteration only uses T[i] in Step 2 and then
			// discards it in Step 6.
			d := bLimbs[i]
			c1 := addMulVVW(T[i:n+i], aLimbs, d)

			// Step 6 is replaced by shifting the virtual window we operate
			// over: T of the algorithm is T[i:] for us. That means that T1 in
			// Step 2 (T mod 2^_W) is simply T[i]. k0 in Step 3 is our m0inv.
			Y := T[i] * m.m0inv

			// Step 4 and 5 add Y × m to T, which as mentioned above is stored
			// at T[i:]. The two carries (from a × d and Y × m) are added up in
			// the next word T[n+i], and the carry bit from that addition is
			// brought forward to the next iteration.
			c2 := addMulVVW(T[i:n+i], mLimbs, Y)
			T[n+i], c = bits.Add(c1, c2, c)
		}

		// Finally for Step 7 we copy the final T window into x, and subtract m
		// if necessary (which as explained in maybeSubtractModulus can be the
		// case both if x >= m, or if x overflowed).
		//
		// The paper suggests in Section 4 that we can do an "Almost Montgomery
		// Multiplication" by subtracting only in the overflow case, but the
		// cost is very similar since the constant time subtraction tells us if
		// x >= m as a side effect, and taking care of the broken invariant is
		// highly undesirable (see https://go.dev/issue/13907).
		copy(x.reset(n).limbs, T[n:])
		x.maybeSubtractModulus(choice(c), m)

	// The following specialized cases follow the exact same algorithm, but
	// optimized for the sizes most used in RSA. addMulVVW is implemented in
	// assembly with loop unrolling depending on the architecture and bounds
	// checks are removed by the compiler thanks to the constant size.
	case 1024 / _W:
		const n = 1024 / _W // compiler hint
		T := make([]uint, n*2)
		var c uint
		for i := 0; i < n; i++ {
			d := bLimbs[i]
			c1 := addMulVVW1024(&T[i], &aLimbs[0], d)
			Y := T[i] * m.m0inv
			c2 := addMulVVW1024(&T[i], &mLimbs[0], Y)
			T[n+i], c = bits.Add(c1, c2, c)
		}
		copy(x.reset(n).limbs, T[n:])
		x.maybeSubtractModulus(choice(c), m)

	case 1536 / _W:
		const n = 1536 / _W // compiler hint
		T := make([]uint, n*2)
		var c uint
		for i := 0; i < n; i++ {
			d := bLimbs[i]
			c1 := addMulVVW1536(&T[i], &aLimbs[0], d)
			Y := T[i] * m.m0inv
			c2 := addMulVVW1536(&T[i], &mLimbs[0], Y)
			T[n+i], c = bits.Add(c1, c2, c)
		}
		copy(x.reset(n).limbs, T[n:])
		x.maybeSubtractModulus(choice(c), m)

	case 2048 / _W:
		const n = 2048 / _W // compiler hint
		T := make([]uint, n*2)
		var c uint
		for i := 0; i < n; i++ {
			d := bLimbs[i]
			c1 := addMulVVW2048(&T[i], &aLimbs[0], d)
			Y := T[i] * m.m0inv
			c2 := addMulVVW2048(&T[i], &mLimbs[0], Y)
			T[n+i], c = bits.Add(c1, c2, c)
		}
		copy(x.reset(n).limbs, T[n:])
		x.maybeSubtractModulus(choice(c), m)
	}

	return x
}

// addMulVVW multiplies the multi-word value x by the single-word value y,
// adding the result to the multi-word value z and returning the final carry.
// It can be thought of as one row of a pen-and-paper column multiplication.
func addMulVVW(z, x []uint, y uint) (carry uint) {
	_ = x[len(z)-1] // bounds check elimination hint
	for i := range z {
		hi, lo := bits.Mul(x[i], y)
		lo, c := bits.Add(lo, z[i], 0)
		// We use bits.Add with zero to get an add-with-carry instruction that
		// absorbs the carry from the previous bits.Add.
		hi, _ = bits.Add(hi, 0, c)
		lo, c = bits.Add(lo, carry, 0)
		hi, _ = bits.Add(hi, 0, c)
		carry = hi
		z[i] = lo
	}
	return carry
}

// Mul calculates x = x * y mod m.
//
// The length of both operands must be the same as the modulus. Both operands
// must already be reduced modulo m.
func (x *Nat) Mul(y *Nat, m *Modulus) *Nat {
	if m.odd {
		// A Montgomery multiplication by a value out of the Montgomery domain
		// takes the result out of Montgomery representation.
		xR := NewNat().set(x).montgomeryRepresentation(m) // xR = x * R mod m
		return x.montgomeryMul(xR, y, m)                  // x = xR * y / R mod m
	}

	n := len(m.nat.limbs)
	xLimbs := x.limbs[:n]
	yLimbs := y.limbs[:n]

	switch n {
	default:
		// Attempt to use a stack-allocated backing array.
		T := make([]uint, 0, preallocLimbs*2)
		if cap(T) < n*2 {
			T = make([]uint, 0, n*2)
		}
		T = T[:n*2]

		// T = x * y
		for i := 0; i < n; i++ {
			T[n+i] = addMulVVW(T[i:n+i], xLimbs, yLimbs[i])
		}

		// x = T mod m
		return x.Mod(&Nat{limbs: T}, m)

	// The following specialized cases follow the exact same algorithm, but
	// optimized for the sizes most used in RSA. See montgomeryMul for details.
	case 1024 / _W:
		const n = 1024 / _W // compiler hint
		T := make([]uint, n*2)
		for i := 0; i < n; i++ {
			T[n+i] = addMulVVW1024(&T[i], &xLimbs[0], yLimbs[i])
		}
		return x.Mod(&Nat{limbs: T}, m)
	case 1536 / _W:
		const n = 1536 / _W // compiler hint
		T := make([]uint, n*2)
		for i := 0; i < n; i++ {
			T[n+i] = addMulVVW1536(&T[i], &xLimbs[0], yLimbs[i])
		}
		return x.Mod(&Nat{limbs: T}, m)
	case 2048 / _W:
		const n = 2048 / _W // compiler hint
		T := make([]uint, n*2)
		for i := 0; i < n; i++ {
			T[n+i] = addMulVVW2048(&T[i], &xLimbs[0], yLimbs[i])
		}
		return x.Mod(&Nat{limbs: T}, m)
	}
}

// Exp calculates out = x^e mod m.
//
// The exponent e is represented in big-endian order. The output will be resized
// to the size of m and overwritten. x must already be reduced modulo m.
//
// m must be odd, or Exp will panic.
func (out *Nat) Exp(x *Nat, e []byte, m *Modulus) *Nat {
	if !m.odd {
		panic("bigmod: modulus for Exp must be odd")
	}

	// We use a 4 bit window. For our RSA workload, 4 bit windows are faster
	// than 2 bit windows, but use an extra 12 nats worth of scratch space.
	// Using bit sizes that don't divide 8 are more complex to implement, but
	// are likely to be more efficient if necessary.

	table := [(1 << 4) - 1]*Nat{ // table[i] = x ^ (i+1)
		// newNat calls are unrolled so they are allocated on the stack.
		NewNat(), NewNat(), NewNat(), NewNat(), NewNat(),
		NewNat(), NewNat(), NewNat(), NewNat(), NewNat(),
		NewNat(), NewNat(), NewNat(), NewNat(), NewNat(),
	}
	table[0].set(x).montgomeryRepresentation(m)
	for i := 1; i < len(table); i++ {
		table[i].montgomeryMul(table[i-1], table[0], m)
	}

	out.resetFor(m)
	out.limbs[0] = 1
	out.montgomeryRepresentation(m)
	tmp := NewNat().ExpandFor(m)
	for _, b := range e {
		for _, j := range []int{4, 0} {
			// Square four times. Optimization note: this can be implemented
			// more efficiently than with generic Montgomery multiplication.
			out.montgomeryMul(out, out, m)
			out.montgomeryMul(out, out, m)
			out.montgomeryMul(out, out, m)
			out.montgomeryMul(out, out, m)

			// Select x^k in constant time from the table.
			k := uint((b >> j) & 0b1111)
			for i := range table {
				tmp.assign(ctEq(k, uint(i+1)), table[i])
			}

			// Multiply by x^k, discarding the result if k = 0.
			tmp.montgomeryMul(out, tmp, m)
			out.assign(not(ctEq(k, 0)), tmp)
		}
	}

	return out.montgomeryReduction(m)
}

// ExpShortVarTime calculates out = x^e mod m.
//
// The output will be resized to the size of m and overwritten. x must already
// be reduced modulo m. This leaks the exponent through timing side-channels.
//
// m must be odd, or ExpShortVarTime will panic.
func (out *Nat) ExpShortVarTime(x *Nat, e uint, m *Modulus) *Nat {
	if !m.odd {
		panic("bigmod: modulus for ExpShortVarTime must be odd")
	}
	// For short exponents, precomputing a table and using a window like in Exp
	// doesn't pay off. Instead, we do a simple conditional square-and-multiply
	// chain, skipping the initial run of zeroes.
	xR := NewNat().set(x).montgomeryRepresentation(m)
	out.set(xR)
	for i := bits.UintSize - bitLen(e) + 1; i < bits.UintSize; i++ {
		out.montgomeryMul(out, out, m)
		if k := (e >> (bits.UintSize - i - 1)) & 1; k != 0 {
			out.montgomeryMul(out, xR, m)
		}
	}
	return out.montgomeryReduction(m)
}

// InverseVarTime calculates x = a⁻¹ mod m and returns (x, true) if a is
// invertible. Otherwise, InverseVarTime returns (x, false) and x is not
// modified.
//
// a must be reduced modulo m, but doesn't need to have the same size. The
// output will be resized to the size of m and overwritten.
func (x *Nat) InverseVarTime(a *Nat, m *Modulus) (*Nat, bool) {
	// This is the extended binary GCD algorithm described in the Handbook of
	// Applied Cryptography, Algorithm 14.61, adapted by BoringSSL to bound
	// coefficients and avoid negative numbers. For more details and proof of
	// correctness, see https://github.com/mit-plv/fiat-crypto/pull/333/files.
	//
	// Following the proof linked in the PR above, the changes are:
	//
	// 1. Negate [B] and [C] so they are positive. The invariant now involves a
	//    subtraction.
	// 2. If step 2 (both [x] and [y] are even) runs, abort immediately. This
	//    algorithm only cares about [x] and [y] relatively prime.
	// 3. Subtract copies of [x] and [y] as needed in step 6 (both [u] and [v]
	//    are odd) so coefficients stay in bounds.
	// 4. Replace the [u >= v] check with [u > v]. This changes the end
	//    condition to [v = 0] rather than [u = 0]. This saves an extra
	//    subtraction due to which coefficients were negated.
	// 5. Rename x and y to a and n, to capture that one is a modulus.
	// 6. Rearrange steps 4 through 6 slightly. Merge the loops in steps 4 and
	//    5 into the main loop (step 7's goto), and move step 6 to the start of
	//    the loop iteration, ensuring each loop iteration halves at least one
	//    value.
	//
	// Note this algorithm does not handle either input being zero.

	if a.IsZero() == yes {
		return x, false
	}
	if a.IsOdd() == no && !m.odd {
		// a and m are not coprime, as they are both even.
		return x, false
	}

	u := NewNat().set(a).ExpandFor(m)
	v := m.Nat()

	A := NewNat().reset(len(m.nat.limbs))
	A.limbs[0] = 1
	B := NewNat().reset(len(a.limbs))
	C := NewNat().reset(len(m.nat.limbs))
	D := NewNat().reset(len(a.limbs))
	D.limbs[0] = 1

	// Before and after each loop iteration, the following hold:
	//
	//   u = A*a - B*m
	//   v = D*m - C*a
	//   0 < u <= a
	//   0 <= v <= m
	//   0 <= A < m
	//   0 <= B <= a
	//   0 <= C < m
	//   0 <= D <= a
	//
	// After each loop iteration, u and v only get smaller, and at least one of
	// them shrinks by at least a factor of two.
	for {
		// If both u and v are odd, subtract the smaller from the larger.
		// If u = v, we need to subtract from v to hit the modified exit condition.
		if u.IsOdd() == yes && v.IsOdd() == yes {
			if v.cmpGeq(u) == no {
				u.sub(v)
				A.Add(C, m)
				B.Add(D, &Modulus{nat: a})
			} else {
				v.sub(u)
				C.Add(A, m)
				D.Add(B, &Modulus{nat: a})
			}
		}

		// Exactly one of u and v is now even.
		if u.IsOdd() == v.IsOdd() {
			panic("bigmod: internal error: u and v are not in the expected state")
		}

		// Halve the even one and adjust the corresponding coefficient.
		if u.IsOdd() == no {
			rshift1(u, 0)
			if A.IsOdd() == yes || B.IsOdd() == yes {
				rshift1(A, A.add(m.nat))
				rshift1(B, B.add(a))
			} else {
				rshift1(A, 0)
				rshift1(B, 0)
			}
		} else { // v.IsOdd() == no
			rshift1(v, 0)
			if C.IsOdd() == yes || D.IsOdd() == yes {
				rshift1(C, C.add(m.nat))
				rshift1(D, D.add(a))
			} else {
				rshift1(C, 0)
				rshift1(D, 0)
			}
		}

		if v.IsZero() == yes {
			if u.IsOne() == no {
				return x, false
			}
			return x.set(A), true
		}
	}
}

func rshift1(a *Nat, carry uint) {
	size := len(a.limbs)
	aLimbs := a.limbs[:size]

	for i := range size {
		aLimbs[i] >>= 1
		if i+1 < size {
			aLimbs[i] |= aLimbs[i+1] << (_W - 1)
		} else {
			aLimbs[i] |= carry << (_W - 1)
		}
	}
}

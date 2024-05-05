// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bigmod

import (
	"errors"
	"internal/binary"
	"math/big"
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

// set assigns x = y, optionally resizing x to the appropriate size.
func (x *Nat) set(y *Nat) *Nat {
	x.reset(len(y.limbs))
	copy(x.limbs, y.limbs)
	return x
}

// setBig assigns x = n, optionally resizing n to the appropriate size.
//
// The announced length of x is set based on the actual bit size of the input,
// ignoring leading zeroes.
func (x *Nat) setBig(n *big.Int) *Nat {
	limbs := n.Bits()
	x.reset(len(limbs))
	for i := range limbs {
		x.limbs[i] = uint(limbs[i])
	}
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
	if err := x.setBytes(b, m); err != nil {
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
	if err := x.setBytes(b, m); err != nil {
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
		return uint(binary.BigEndian.Uint64(buf))
	}
	return uint(binary.BigEndian.Uint32(buf))
}

func (x *Nat) setBytes(b []byte, m *Modulus) error {
	x.resetFor(m)
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

// Modulus is used for modular arithmetic, precomputing relevant constants.
//
// Moduli are assumed to be odd numbers. Moduli can also leak the exact
// number of bits needed to store their value, and are stored without padding.
//
// Their actual value is still kept secret.
type Modulus struct {
	// The underlying natural number for this modulus.
	//
	// This will be stored without any padding, and shouldn't alias with any
	// other natural number being used.
	nat     *Nat
	leading int  // number of leading zeros in the modulus
	m0inv   uint // -nat.limbs[0]⁻¹ mod _W
	rr      *Nat // R*R for montgomeryRepresentation
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

// NewModulusFromBig creates a new Modulus from a [big.Int].
//
// The Int must be odd. The number of significant bits (and nothing else) is
// leaked through timing side-channels.
func NewModulusFromBig(n *big.Int) (*Modulus, error) {
	if b := n.Bits(); len(b) == 0 {
		return nil, errors.New("modulus must be >= 0")
	} else if b[0]&1 != 1 {
		return nil, errors.New("modulus must be odd")
	}
	m := &Modulus{}
	m.nat = NewNat().setBig(n)
	m.leading = _W - bitLen(m.nat.limbs[len(m.nat.limbs)-1])
	m.m0inv = minusInverseModW(m.nat.limbs[0])
	m.rr = rr(m)
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

// Nat returns m as a Nat. The return value must not be written to.
func (m *Modulus) Nat() *Nat {
	return m.nat
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
	// A Montgomery multiplication by a value out of the Montgomery domain
	// takes the result out of Montgomery representation.
	xR := NewNat().set(x).montgomeryRepresentation(m) // xR = x * R mod m
	return x.montgomeryMul(xR, y, m)                  // x = xR * y / R mod m
}

// Exp calculates out = x^e mod m.
//
// The exponent e is represented in big-endian order. The output will be resized
// to the size of m and overwritten. x must already be reduced modulo m.
func (out *Nat) Exp(x *Nat, e []byte, m *Modulus) *Nat {
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
func (out *Nat) ExpShortVarTime(x *Nat, e uint, m *Modulus) *Nat {
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

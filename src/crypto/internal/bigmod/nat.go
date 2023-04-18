// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bigmod

import (
	"errors"
	"math/big"
	"math/bits"
)

const (
	// _W is the number of bits we use for our limbs.
	_W = bits.UintSize - 1
	// _MASK selects _W bits from a full machine word.
	_MASK = (1 << _W) - 1
)

// choice represents a constant-time boolean. The value of choice is always
// either 1 or 0. We use an int instead of bool in order to make decisions in
// constant time by turning it into a mask.
type choice uint

func not(c choice) choice { return 1 ^ c }

const yes = choice(1)
const no = choice(0)

// ctSelect returns x if on == 1, and y if on == 0. The execution time of this
// function does not depend on its inputs. If on is any value besides 1 or 0,
// the result is undefined.
func ctSelect(on choice, x, y uint) uint {
	// When on == 1, mask is 0b111..., otherwise mask is 0b000...
	mask := -uint(on)
	// When mask is all zeros, we just have y, otherwise, y cancels with itself.
	return y ^ (mask & (y ^ x))
}

// ctEq returns 1 if x == y, and 0 otherwise. The execution time of this
// function does not depend on its inputs.
func ctEq(x, y uint) choice {
	// If x != y, then either x - y or y - x will generate a carry.
	_, c1 := bits.Sub(x, y, 0)
	_, c2 := bits.Sub(y, x, 0)
	return not(choice(c1 | c2))
}

// ctGeq returns 1 if x >= y, and 0 otherwise. The execution time of this
// function does not depend on its inputs.
func ctGeq(x, y uint) choice {
	// If x < y, then x - y generates a carry.
	_, carry := bits.Sub(x, y, 0)
	return not(choice(carry))
}

// Nat represents an arbitrary natural number
//
// Each Nat has an announced length, which is the number of limbs it has stored.
// Operations on this number are allowed to leak this length, but will not leak
// any information about the values contained in those limbs.
type Nat struct {
	// limbs is a little-endian representation in base 2^W with
	// W = bits.UintSize - 1. The top bit is always unset between operations.
	//
	// The top bit is left unset to optimize Montgomery multiplication, in the
	// inner loop of exponentiation. Using fully saturated limbs would leave us
	// working with 129-bit numbers on 64-bit platforms, wasting a lot of space,
	// and thus time.
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
	for i := range extraLimbs {
		extraLimbs[i] = 0
	}
	x.limbs = x.limbs[:n]
	return x
}

// reset returns a zero nat of n limbs, reusing x's storage if n <= cap(x.limbs).
func (x *Nat) reset(n int) *Nat {
	if cap(x.limbs) < n {
		x.limbs = make([]uint, n)
		return x
	}
	for i := range x.limbs {
		x.limbs[i] = 0
	}
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
	requiredLimbs := (n.BitLen() + _W - 1) / _W
	x.reset(requiredLimbs)

	outI := 0
	shift := 0
	limbs := n.Bits()
	for i := range limbs {
		xi := uint(limbs[i])
		x.limbs[outI] |= (xi << shift) & _MASK
		outI++
		if outI == requiredLimbs {
			return x
		}
		x.limbs[outI] = xi >> (_W - shift)
		shift++ // this assumes bits.UintSize - _W = 1
		if shift == _W {
			shift = 0
			outI++
		}
	}
	return x
}

// Bytes returns x as a zero-extended big-endian byte slice. The size of the
// slice will match the size of m.
//
// x must have the same size as m and it must be reduced modulo m.
func (x *Nat) Bytes(m *Modulus) []byte {
	bytes := make([]byte, m.Size())
	shift := 0
	outI := len(bytes) - 1
	for _, limb := range x.limbs {
		remainingBits := _W
		for remainingBits >= 8 {
			bytes[outI] |= byte(limb) << shift
			consumed := 8 - shift
			limb >>= consumed
			remainingBits -= consumed
			shift = 0
			outI--
			if outI < 0 {
				return bytes
			}
		}
		bytes[outI] = byte(limb)
		shift = remainingBits
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

// SetOverflowingBytes assigns x = b, where b is a slice of big-endian bytes. SetOverflowingBytes
// returns an error if b has a longer bit length than m, but reduces overflowing
// values up to 2^⌈log2(m)⌉ - 1.
//
// The output will be resized to the size of m and overwritten.
func (x *Nat) SetOverflowingBytes(b []byte, m *Modulus) (*Nat, error) {
	if err := x.setBytes(b, m); err != nil {
		return nil, err
	}
	leading := _W - bitLen(x.limbs[len(x.limbs)-1])
	if leading < m.leading {
		return nil, errors.New("input overflows the modulus")
	}
	x.sub(x.cmpGeq(m.nat), m.nat)
	return x, nil
}

func (x *Nat) setBytes(b []byte, m *Modulus) error {
	outI := 0
	shift := 0
	x.resetFor(m)
	for i := len(b) - 1; i >= 0; i-- {
		bi := b[i]
		x.limbs[outI] |= uint(bi) << shift
		shift += 8
		if shift >= _W {
			shift -= _W
			x.limbs[outI] &= _MASK
			overflow := bi >> (8 - shift)
			outI++
			if outI >= len(x.limbs) {
				if overflow > 0 || i > 0 {
					return errors.New("input overflows the modulus")
				}
				break
			}
			x.limbs[outI] = uint(overflow)
		}
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
		c = (xLimbs[i] - yLimbs[i] - c) >> _W
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

	for i := 0; i < size; i++ {
		xLimbs[i] = ctSelect(on, yLimbs[i], xLimbs[i])
	}
	return x
}

// add computes x += y if on == 1, and does nothing otherwise. It returns the
// carry of the addition regardless of on.
//
// Both operands must have the same announced length.
func (x *Nat) add(on choice, y *Nat) (c uint) {
	// Eliminate bounds checks in the loop.
	size := len(x.limbs)
	xLimbs := x.limbs[:size]
	yLimbs := y.limbs[:size]

	for i := 0; i < size; i++ {
		res := xLimbs[i] + yLimbs[i] + c
		xLimbs[i] = ctSelect(on, res&_MASK, xLimbs[i])
		c = res >> _W
	}
	return
}

// sub computes x -= y if on == 1, and does nothing otherwise. It returns the
// borrow of the subtraction regardless of on.
//
// Both operands must have the same announced length.
func (x *Nat) sub(on choice, y *Nat) (c uint) {
	// Eliminate bounds checks in the loop.
	size := len(x.limbs)
	xLimbs := x.limbs[:size]
	yLimbs := y.limbs[:size]

	for i := 0; i < size; i++ {
		res := xLimbs[i] - yLimbs[i] - c
		xLimbs[i] = ctSelect(on, res&_MASK, xLimbs[i])
		c = res >> _W
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
	// R*R is 2^(2 * _W * n). We can safely get 2^(_W * (n - 1)) by setting the
	// most significant limb to 1. We then get to R*R by shifting left by _W
	// n + 1 times.
	n := len(rr.limbs)
	rr.limbs[n-1] = 1
	for i := n - 1; i < 2*n; i++ {
		rr.shiftIn(0, m) // x = x * 2^_W mod m
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
	// for 61 bits (and wastes only one iteration for 31 bits).
	//
	// See https://crypto.stackexchange.com/a/47496.
	y := x
	for i := 0; i < 5; i++ {
		y = y * (2 - x*y)
	}
	return (1 << _W) - (y & _MASK)
}

// NewModulusFromBig creates a new Modulus from a [big.Int].
//
// The Int must be odd. The number of significant bits must be leakable.
func NewModulusFromBig(n *big.Int) *Modulus {
	m := &Modulus{}
	m.nat = NewNat().setBig(n)
	m.leading = _W - bitLen(m.nat.limbs[len(m.nat.limbs)-1])
	m.m0inv = minusInverseModW(m.nat.limbs[0])
	m.rr = rr(m)
	return m
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
// This assumes that x is already reduced mod m, and that y < 2^_W.
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
	// based on whether the subtraction underflowed.
	needSubtraction := no
	for i := _W - 1; i >= 0; i-- {
		carry := (y >> i) & 1
		var borrow uint
		for i := 0; i < size; i++ {
			l := ctSelect(needSubtraction, dLimbs[i], xLimbs[i])

			res := l<<1 + carry
			xLimbs[i] = res & _MASK
			carry = res >> _W

			res = xLimbs[i] - mLimbs[i] - borrow
			dLimbs[i] = res & _MASK
			borrow = res >> _W
		}
		// See Add for how carry (aka overflow), borrow (aka underflow), and
		// needSubtraction relate.
		needSubtraction = ctEq(carry, borrow)
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

// ExpandFor ensures out has the right size to work with operations modulo m.
//
// The announced size of out must be smaller than or equal to that of m.
func (out *Nat) ExpandFor(m *Modulus) *Nat {
	return out.expand(len(m.nat.limbs))
}

// resetFor ensures out has the right size to work with operations modulo m.
//
// out is zeroed and may start at any size.
func (out *Nat) resetFor(m *Modulus) *Nat {
	return out.reset(len(m.nat.limbs))
}

// Sub computes x = x - y mod m.
//
// The length of both operands must be the same as the modulus. Both operands
// must already be reduced modulo m.
func (x *Nat) Sub(y *Nat, m *Modulus) *Nat {
	underflow := x.sub(yes, y)
	// If the subtraction underflowed, add m.
	x.add(choice(underflow), m.nat)
	return x
}

// Add computes x = x + y mod m.
//
// The length of both operands must be the same as the modulus. Both operands
// must already be reduced modulo m.
func (x *Nat) Add(y *Nat, m *Modulus) *Nat {
	overflow := x.add(yes, y)
	underflow := not(x.cmpGeq(m.nat)) // x < m

	// Three cases are possible:
	//
	//   - overflow = 0, underflow = 0
	//
	// In this case, addition fits in our limbs, but we can still subtract away
	// m without an underflow, so we need to perform the subtraction to reduce
	// our result.
	//
	//   - overflow = 0, underflow = 1
	//
	// The addition fits in our limbs, but we can't subtract m without
	// underflowing. The result is already reduced.
	//
	//   - overflow = 1, underflow = 1
	//
	// The addition does not fit in our limbs, and the subtraction's borrow
	// would cancel out with the addition's carry. We need to subtract m to
	// reduce our result.
	//
	// The overflow = 1, underflow = 0 case is not possible, because y is at
	// most m - 1, and if adding m - 1 overflows, then subtracting m must
	// necessarily underflow.
	needSubtraction := ctEq(overflow, uint(underflow))

	x.sub(needSubtraction, m.nat)
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
	return x.montgomeryMul(NewNat().set(x), m.rr, m)
}

// montgomeryReduction calculates x = x / R mod m, with R = 2^(_W * n) and
// n = len(m.nat.limbs).
//
// This assumes that x is already reduced mod m.
func (x *Nat) montgomeryReduction(m *Modulus) *Nat {
	// By Montgomery multiplying with 1 not in Montgomery representation, we
	// convert out back from Montgomery representation, because it works out to
	// dividing by R.
	t0 := NewNat().set(x)
	t1 := NewNat().ExpandFor(m)
	t1.limbs[0] = 1
	return x.montgomeryMul(t0, t1, m)
}

// montgomeryMul calculates d = a * b / R mod m, with R = 2^(_W * n) and
// n = len(m.nat.limbs), using the Montgomery Multiplication technique.
//
// All inputs should be the same length, not aliasing d, and already
// reduced modulo m. d will be resized to the size of m and overwritten.
func (d *Nat) montgomeryMul(a *Nat, b *Nat, m *Modulus) *Nat {
	d.resetFor(m)
	if len(a.limbs) != len(m.nat.limbs) || len(b.limbs) != len(m.nat.limbs) {
		panic("bigmod: invalid montgomeryMul input")
	}

	// See https://bearssl.org/bigint.html#montgomery-reduction-and-multiplication
	// for a description of the algorithm implemented mostly in montgomeryLoop.
	// See Add for how overflow, underflow, and needSubtraction relate.
	overflow := montgomeryLoop(d.limbs, a.limbs, b.limbs, m.nat.limbs, m.m0inv)
	underflow := not(d.cmpGeq(m.nat)) // d < m
	needSubtraction := ctEq(overflow, uint(underflow))
	d.sub(needSubtraction, m.nat)

	return d
}

func montgomeryLoopGeneric(d, a, b, m []uint, m0inv uint) (overflow uint) {
	// Eliminate bounds checks in the loop.
	size := len(d)
	a = a[:size]
	b = b[:size]
	m = m[:size]

	for _, ai := range a {
		// This is an unrolled iteration of the loop below with j = 0.
		hi, lo := bits.Mul(ai, b[0])
		z_lo, c := bits.Add(d[0], lo, 0)
		f := (z_lo * m0inv) & _MASK // (d[0] + a[i] * b[0]) * m0inv
		z_hi, _ := bits.Add(0, hi, c)
		hi, lo = bits.Mul(f, m[0])
		z_lo, c = bits.Add(z_lo, lo, 0)
		z_hi, _ = bits.Add(z_hi, hi, c)
		carry := z_hi<<1 | z_lo>>_W

		for j := 1; j < size; j++ {
			// z = d[j] + a[i] * b[j] + f * m[j] + carry <= 2^(2W+1) - 2^(W+1) + 2^W
			hi, lo := bits.Mul(ai, b[j])
			z_lo, c := bits.Add(d[j], lo, 0)
			z_hi, _ := bits.Add(0, hi, c)
			hi, lo = bits.Mul(f, m[j])
			z_lo, c = bits.Add(z_lo, lo, 0)
			z_hi, _ = bits.Add(z_hi, hi, c)
			z_lo, c = bits.Add(z_lo, carry, 0)
			z_hi, _ = bits.Add(z_hi, 0, c)
			d[j-1] = z_lo & _MASK
			carry = z_hi<<1 | z_lo>>_W // carry <= 2^(W+1) - 2
		}

		z := overflow + carry // z <= 2^(W+1) - 1
		d[size-1] = z & _MASK
		overflow = z >> _W // overflow <= 1
	}
	return
}

// Mul calculates x *= y mod m.
//
// x and y must already be reduced modulo m, they must share its announced
// length, and they may not alias.
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
	// Using bit sizes that don't divide 8 are more complex to implement.

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
	t0 := NewNat().ExpandFor(m)
	t1 := NewNat().ExpandFor(m)
	for _, b := range e {
		for _, j := range []int{4, 0} {
			// Square four times.
			t1.montgomeryMul(out, out, m)
			out.montgomeryMul(t1, t1, m)
			t1.montgomeryMul(out, out, m)
			out.montgomeryMul(t1, t1, m)

			// Select x^k in constant time from the table.
			k := uint((b >> j) & 0b1111)
			for i := range table {
				t0.assign(ctEq(k, uint(i+1)), table[i])
			}

			// Multiply by x^k, discarding the result if k = 0.
			t1.montgomeryMul(out, t0, m)
			out.assign(not(ctEq(k, 0)), t1)
		}
	}

	return out.montgomeryReduction(m)
}

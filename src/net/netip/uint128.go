// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package netip

import "math/bits"

// uint128 represents a uint128 using two uint64s.
//
// When the methods below mention a bit number, bit 0 is the most
// significant bit (in hi) and bit 127 is the lowest (lo&1).
type uint128 struct {
	hi uint64
	lo uint64
}

// mask6 returns a uint128 bitmask with the topmost n bits of a
// 128-bit number.
func mask6(n int) uint128 {
	return uint128{^(^uint64(0) >> n), ^uint64(0) << (128 - n)}
}

// isZero reports whether u == 0.
//
// It's faster than u == (uint128{}) because the compiler (as of Go
// 1.15/1.16b1) doesn't do this trick and instead inserts a branch in
// its eq alg's generated code.
func (u uint128) isZero() bool { return u.hi|u.lo == 0 }

// and returns the bitwise AND of u and m (u&m).
func (u uint128) and(m uint128) uint128 {
	return uint128{u.hi & m.hi, u.lo & m.lo}
}

// xor returns the bitwise XOR of u and m (u^m).
func (u uint128) xor(m uint128) uint128 {
	return uint128{u.hi ^ m.hi, u.lo ^ m.lo}
}

// or returns the bitwise OR of u and m (u|m).
func (u uint128) or(m uint128) uint128 {
	return uint128{u.hi | m.hi, u.lo | m.lo}
}

// not returns the bitwise NOT of u.
func (u uint128) not() uint128 {
	return uint128{^u.hi, ^u.lo}
}

// subOne returns u - 1.
func (u uint128) subOne() uint128 {
	lo, borrow := bits.Sub64(u.lo, 1, 0)
	return uint128{u.hi - borrow, lo}
}

// addOne returns u + 1.
func (u uint128) addOne() uint128 {
	lo, carry := bits.Add64(u.lo, 1, 0)
	return uint128{u.hi + carry, lo}
}

func u64CommonPrefixLen(a, b uint64) uint8 {
	return uint8(bits.LeadingZeros64(a ^ b))
}

func (u uint128) commonPrefixLen(v uint128) (n uint8) {
	if n = u64CommonPrefixLen(u.hi, v.hi); n == 64 {
		n += u64CommonPrefixLen(u.lo, v.lo)
	}
	return
}

// halves returns the two uint64 halves of the uint128.
//
// Logically, think of it as returning two uint64s.
// It only returns pointers for inlining reasons on 32-bit platforms.
func (u *uint128) halves() [2]*uint64 {
	return [2]*uint64{&u.hi, &u.lo}
}

// bitsSetFrom returns a copy of u with the given bit
// and all subsequent ones set.
func (u uint128) bitsSetFrom(bit uint8) uint128 {
	return u.or(mask6(int(bit)).not())
}

// bitsClearedFrom returns a copy of u with the given bit
// and all subsequent ones cleared.
func (u uint128) bitsClearedFrom(bit uint8) uint128 {
	return u.and(mask6(int(bit)))
}

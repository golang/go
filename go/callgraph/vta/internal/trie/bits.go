// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trie

import (
	"math/bits"
)

// This file contains bit twiddling functions for Patricia tries.
// Consult this paper for details.
//   C. Okasaki and A. Gill, “Fast mergeable integer maps,” in ACM SIGPLAN
//   Workshop on ML, September 1998, pp. 77–86.

// key is a key in a Map.
type key uint64

// bitpos is the position of a bit. A position is represented by having a 1
// bit in that position.
// Examples:
// * 0b0010 is the position of the `1` bit in 2.
//    It is the 3rd most specific bit position in big endian encoding
//    (0b0 and 0b1 are more specific).
// * 0b0100 is the position of the bit that 1 and 5 disagree on.
// * 0b0 is a special value indicating that all bit agree.
type bitpos uint64

// prefixes represent a set of keys that all agree with the
// prefix up to a bitpos m.
//
// The value for a prefix is determined by the mask(k, m) function.
// (See mask for details on the values.)
// A `p` prefix for position `m` matches a key `k` iff mask(k, m) == p.
// A prefix always mask(p, m) == p.
//
// A key is its own prefix for the bit position 64,
//   e.g. seeing a `prefix(key)` is not a problem.
// Prefixes should never be turned into keys.
type prefix uint64

// branchingBit returns the position of the first bit in `x` and `y`
// that are not equal.
func branchingBit(x, y prefix) bitpos {
	p := x ^ y
	if p == 0 {
		return 0
	}
	return bitpos(1) << uint(bits.Len64(uint64(p))-1) // uint conversion needed for go1.12
}

// zeroBit returns true if k has a 0 bit at position `b`.
func zeroBit(k prefix, b bitpos) bool {
	return (uint64(k) & uint64(b)) == 0
}

// matchPrefix returns true if a prefix k matches a prefix p up to position `b`.
func matchPrefix(k prefix, p prefix, b bitpos) bool {
	return mask(k, b) == p
}

// mask returns a prefix of `k` with all bits after and including `b` zeroed out.
//
// In big endian encoding, this value is the [64-(m-1)] most significant bits of k
// followed by a `0` bit at bitpos m, followed m-1 `1` bits.
// Examples:
//  prefix(0b1011) for a bitpos 0b0100 represents the keys:
//    0b1000, 0b1001, 0b1010, 0b1011, 0b1100, 0b1101, 0b1110, 0b1111
//
// This mask function has the property that if matchPrefix(k, p, b), then
// k <= p if and only if zeroBit(k, m). This induces binary search tree tries.
// See Okasaki & Gill for more details about this choice of mask function.
//
// mask is idempotent for a given `b`, i.e. mask(mask(p, b), b) == mask(p,b).
func mask(k prefix, b bitpos) prefix {
	return prefix((uint64(k) | (uint64(b) - 1)) & (^uint64(b)))
}

// ord returns true if m comes before n in the bit ordering.
func ord(m, n bitpos) bool {
	return m > n // big endian encoding
}

// prefixesOverlap returns true if there is some key a prefix `p` for bitpos `m`
// can hold that can also be held by a prefix `q` for some bitpos `n`.
//
// This is equivalent to:
//   m ==n && p == q,
//   higher(m, n) && matchPrefix(q, p, m), or
//   higher(n, m) && matchPrefix(p, q, n)
func prefixesOverlap(p prefix, m bitpos, q prefix, n bitpos) bool {
	fbb := n
	if ord(m, n) {
		fbb = m
	}
	return mask(p, fbb) == mask(q, fbb)
	// Lemma:
	//   mask(p, fbb) == mask(q, fbb)
	// iff
	//   m > n && matchPrefix(q, p, m) or  (note: big endian encoding)
	//   m < n && matchPrefix(p, q, n) or  (note: big endian encoding)
	//   m ==n && p == q
	// Quick-n-dirty proof:
	// p == mask(p0, m) for some p0 by precondition.
	// q == mask(q0, n) for some q0 by precondition.
	// So mask(p, m) == p and mask(q, n) == q as mask(*, n') is idempotent.
	//
	// [=> proof]
	// Suppose mask(p, fbb) == mask(q, fbb).
	// if m ==n, p == mask(p, m) == mask(p, fbb) == mask(q, fbb) == mask(q, n) == q
	// if m > n, fbb = firstBranchBit(m, n) = m (big endian).
	//   p == mask(p, m) == mask(p, fbb) == mask(q, fbb) == mask(q, m)
	//   so mask(q, m) == p or matchPrefix(q, p, m)
	// if m < n, is symmetric to the above.
	//
	// [<= proof]
	// case m ==n && p == q. Then mask(p, fbb) == mask(q, fbb)
	//
	// case m > n && matchPrefix(q, p, m).
	// fbb == firstBranchBit(m, n) == m (by m>n).
	// mask(q, fbb) == mask(q, m) == p == mask(p, m) == mask(p, fbb)
	//
	// case m < n && matchPrefix(p, q, n) is symmetric.
}

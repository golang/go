// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package maphash provides hash functions on byte sequences.
// These hash functions are intended to be used to implement hash tables or
// other data structures that need to map arbitrary strings or byte
// sequences to a uniform distribution on unsigned 64-bit integers.
// Each different instance of a hash table or data structure should use its own Seed.
//
// The hash functions are not cryptographically secure.
// (See crypto/sha256 and crypto/sha512 for cryptographic use.)
//
package maphash

import "unsafe"

// A Seed is a random value that selects the specific hash function
// computed by a Hash. If two Hashes use the same Seeds, they
// will compute the same hash values for any given input.
// If two Hashes use different Seeds, they are very likely to compute
// distinct hash values for any given input.
//
// A Seed must be initialized by calling MakeSeed.
// The zero seed is uninitialized and not valid for use with Hash's SetSeed method.
//
// Each Seed value is local to a single process and cannot be serialized
// or otherwise recreated in a different process.
type Seed struct {
	s uint64
}

// A Hash computes a seeded hash of a byte sequence.
//
// The zero Hash is a valid Hash ready to use.
// A zero Hash chooses a random seed for itself during
// the first call to a Reset, Write, Seed, Sum64, or Seed method.
// For control over the seed, use SetSeed.
//
// The computed hash values depend only on the initial seed and
// the sequence of bytes provided to the Hash object, not on the way
// in which the bytes are provided. For example, the three sequences
//
//     h.Write([]byte{'f','o','o'})
//     h.WriteByte('f'); h.WriteByte('o'); h.WriteByte('o')
//     h.WriteString("foo")
//
// all have the same effect.
//
// Hashes are intended to be collision-resistant, even for situations
// where an adversary controls the byte sequences being hashed.
//
// A Hash is not safe for concurrent use by multiple goroutines, but a Seed is.
// If multiple goroutines must compute the same seeded hash,
// each can declare its own Hash and call SetSeed with a common Seed.
type Hash struct {
	_     [0]func() // not comparable
	seed  Seed      // initial seed used for this hash
	state Seed      // current hash of all flushed bytes
	buf   [64]byte  // unflushed byte buffer
	n     int       // number of unflushed bytes
}

// initSeed seeds the hash if necessary.
// initSeed is called lazily before any operation that actually uses h.seed/h.state.
// Note that this does not include Write/WriteByte/WriteString in the case
// where they only add to h.buf. (If they write too much, they call h.flush,
// which does call h.initSeed.)
func (h *Hash) initSeed() {
	if h.seed.s == 0 {
		h.setSeed(MakeSeed())
	}
}

// WriteByte adds b to the sequence of bytes hashed by h.
// It never fails; the error result is for implementing io.ByteWriter.
func (h *Hash) WriteByte(b byte) error {
	if h.n == len(h.buf) {
		h.flush()
	}
	h.buf[h.n] = b
	h.n++
	return nil
}

// Write adds b to the sequence of bytes hashed by h.
// It always writes all of b and never fails; the count and error result are for implementing io.Writer.
func (h *Hash) Write(b []byte) (int, error) {
	size := len(b)
	for h.n+len(b) > len(h.buf) {
		k := copy(h.buf[h.n:], b)
		h.n = len(h.buf)
		b = b[k:]
		h.flush()
	}
	h.n += copy(h.buf[h.n:], b)
	return size, nil
}

// WriteString adds the bytes of s to the sequence of bytes hashed by h.
// It always writes all of s and never fails; the count and error result are for implementing io.StringWriter.
func (h *Hash) WriteString(s string) (int, error) {
	size := len(s)
	for h.n+len(s) > len(h.buf) {
		k := copy(h.buf[h.n:], s)
		h.n = len(h.buf)
		s = s[k:]
		h.flush()
	}
	h.n += copy(h.buf[h.n:], s)
	return size, nil
}

// Seed returns h's seed value.
func (h *Hash) Seed() Seed {
	h.initSeed()
	return h.seed
}

// SetSeed sets h to use seed, which must have been returned by MakeSeed
// or by another Hash's Seed method.
// Two Hash objects with the same seed behave identically.
// Two Hash objects with different seeds will very likely behave differently.
// Any bytes added to h before this call will be discarded.
func (h *Hash) SetSeed(seed Seed) {
	h.setSeed(seed)
	h.n = 0
}

// setSeed sets seed without discarding accumulated data.
func (h *Hash) setSeed(seed Seed) {
	if seed.s == 0 {
		panic("maphash: use of uninitialized Seed")
	}
	h.seed = seed
	h.state = seed
}

// Reset discards all bytes added to h.
// (The seed remains the same.)
func (h *Hash) Reset() {
	h.initSeed()
	h.state = h.seed
	h.n = 0
}

// precondition: buffer is full.
func (h *Hash) flush() {
	if h.n != len(h.buf) {
		panic("maphash: flush of partially full buffer")
	}
	h.initSeed()
	h.state.s = rthash(h.buf[:], h.state.s)
	h.n = 0
}

// Sum64 returns h's current 64-bit value, which depends on
// h's seed and the sequence of bytes added to h since the
// last call to Reset or SetSeed.
//
// All bits of the Sum64 result are close to uniformly and
// independently distributed, so it can be safely reduced
// by using bit masking, shifting, or modular arithmetic.
func (h *Hash) Sum64() uint64 {
	h.initSeed()
	return rthash(h.buf[:h.n], h.state.s)
}

// MakeSeed returns a new random seed.
func MakeSeed() Seed {
	var s1, s2 uint64
	for {
		s1 = uint64(runtime_fastrand())
		s2 = uint64(runtime_fastrand())
		// We use seed 0 to indicate an uninitialized seed/hash,
		// so keep trying until we get a non-zero seed.
		if s1|s2 != 0 {
			break
		}
	}
	return Seed{s: s1<<32 + s2}
}

//go:linkname runtime_fastrand runtime.fastrand
func runtime_fastrand() uint32

func rthash(b []byte, seed uint64) uint64 {
	if len(b) == 0 {
		return seed
	}
	// The runtime hasher only works on uintptr. For 64-bit
	// architectures, we use the hasher directly. Otherwise,
	// we use two parallel hashers on the lower and upper 32 bits.
	if unsafe.Sizeof(uintptr(0)) == 8 {
		return uint64(runtime_memhash(unsafe.Pointer(&b[0]), uintptr(seed), uintptr(len(b))))
	}
	lo := runtime_memhash(unsafe.Pointer(&b[0]), uintptr(seed), uintptr(len(b)))
	hi := runtime_memhash(unsafe.Pointer(&b[0]), uintptr(seed>>32), uintptr(len(b)))
	return uint64(hi)<<32 | uint64(lo)
}

//go:linkname runtime_memhash runtime.memhash
//go:noescape
func runtime_memhash(p unsafe.Pointer, seed, s uintptr) uintptr

// Sum appends the hash's current 64-bit value to b.
// It exists for implementing hash.Hash.
// For direct calls, it is more efficient to use Sum64.
func (h *Hash) Sum(b []byte) []byte {
	x := h.Sum64()
	return append(b,
		byte(x>>0),
		byte(x>>8),
		byte(x>>16),
		byte(x>>24),
		byte(x>>32),
		byte(x>>40),
		byte(x>>48),
		byte(x>>56))
}

// Size returns h's hash value size, 8 bytes.
func (h *Hash) Size() int { return 8 }

// BlockSize returns h's block size.
func (h *Hash) BlockSize() int { return len(h.buf) }

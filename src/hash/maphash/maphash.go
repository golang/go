// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package maphash provides hash functions on byte sequences and comparable values.
// These hash functions are intended to be used to implement hash tables or
// other data structures that need to map arbitrary strings or byte
// sequences to a uniform distribution on unsigned 64-bit integers.
// Each different instance of a hash table or data structure should use its own [Seed].
//
// The hash functions are not cryptographically secure.
// (See crypto/sha256 and crypto/sha512 for cryptographic use.)
package maphash

import (
	"internal/byteorder"
	"math"
)

// A Seed is a random value that selects the specific hash function
// computed by a [Hash]. If two Hashes use the same Seeds, they
// will compute the same hash values for any given input.
// If two Hashes use different Seeds, they are very likely to compute
// distinct hash values for any given input.
//
// A Seed must be initialized by calling [MakeSeed].
// The zero seed is uninitialized and not valid for use with [Hash]'s SetSeed method.
//
// Each Seed value is local to a single process and cannot be serialized
// or otherwise recreated in a different process.
type Seed struct {
	s uint64
}

// Bytes returns the hash of b with the given seed.
//
// Bytes is equivalent to, but more convenient and efficient than:
//
//	var h Hash
//	h.SetSeed(seed)
//	h.Write(b)
//	return h.Sum64()
func Bytes(seed Seed, b []byte) uint64 {
	state := seed.s
	if state == 0 {
		panic("maphash: use of uninitialized Seed")
	}

	if len(b) > bufSize {
		b = b[:len(b):len(b)] // merge len and cap calculations when reslicing
		for len(b) > bufSize {
			state = rthash(b[:bufSize], state)
			b = b[bufSize:]
		}
	}
	return rthash(b, state)
}

// String returns the hash of s with the given seed.
//
// String is equivalent to, but more convenient and efficient than:
//
//	var h Hash
//	h.SetSeed(seed)
//	h.WriteString(s)
//	return h.Sum64()
func String(seed Seed, s string) uint64 {
	state := seed.s
	if state == 0 {
		panic("maphash: use of uninitialized Seed")
	}
	for len(s) > bufSize {
		state = rthashString(s[:bufSize], state)
		s = s[bufSize:]
	}
	return rthashString(s, state)
}

// A Hash computes a seeded hash of a byte sequence.
//
// The zero Hash is a valid Hash ready to use.
// A zero Hash chooses a random seed for itself during
// the first call to a Reset, Write, Seed, or Sum64 method.
// For control over the seed, use SetSeed.
//
// The computed hash values depend only on the initial seed and
// the sequence of bytes provided to the Hash object, not on the way
// in which the bytes are provided. For example, the three sequences
//
//	h.Write([]byte{'f','o','o'})
//	h.WriteByte('f'); h.WriteByte('o'); h.WriteByte('o')
//	h.WriteString("foo")
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
	_     [0]func()     // not comparable
	seed  Seed          // initial seed used for this hash
	state Seed          // current hash of all flushed bytes
	buf   [bufSize]byte // unflushed byte buffer
	n     int           // number of unflushed bytes
}

// bufSize is the size of the Hash write buffer.
// The buffer ensures that writes depend only on the sequence of bytes,
// not the sequence of WriteByte/Write/WriteString calls,
// by always calling rthash with a full buffer (except for the tail).
const bufSize = 128

// initSeed seeds the hash if necessary.
// initSeed is called lazily before any operation that actually uses h.seed/h.state.
// Note that this does not include Write/WriteByte/WriteString in the case
// where they only add to h.buf. (If they write too much, they call h.flush,
// which does call h.initSeed.)
func (h *Hash) initSeed() {
	if h.seed.s == 0 {
		seed := MakeSeed()
		h.seed = seed
		h.state = seed
	}
}

// WriteByte adds b to the sequence of bytes hashed by h.
// It never fails; the error result is for implementing [io.ByteWriter].
func (h *Hash) WriteByte(b byte) error {
	if h.n == len(h.buf) {
		h.flush()
	}
	h.buf[h.n] = b
	h.n++
	return nil
}

// Write adds b to the sequence of bytes hashed by h.
// It always writes all of b and never fails; the count and error result are for implementing [io.Writer].
func (h *Hash) Write(b []byte) (int, error) {
	size := len(b)
	// Deal with bytes left over in h.buf.
	// h.n <= bufSize is always true.
	// Checking it is ~free and it lets the compiler eliminate a bounds check.
	if h.n > 0 && h.n <= bufSize {
		k := copy(h.buf[h.n:], b)
		h.n += k
		if h.n < bufSize {
			// Copied the entirety of b to h.buf.
			return size, nil
		}
		b = b[k:]
		h.flush()
		// No need to set h.n = 0 here; it happens just before exit.
	}
	// Process as many full buffers as possible, without copying, and calling initSeed only once.
	if len(b) > bufSize {
		h.initSeed()
		for len(b) > bufSize {
			h.state.s = rthash(b[:bufSize], h.state.s)
			b = b[bufSize:]
		}
	}
	// Copy the tail.
	copy(h.buf[:], b)
	h.n = len(b)
	return size, nil
}

// WriteString adds the bytes of s to the sequence of bytes hashed by h.
// It always writes all of s and never fails; the count and error result are for implementing [io.StringWriter].
func (h *Hash) WriteString(s string) (int, error) {
	// WriteString mirrors Write. See Write for comments.
	size := len(s)
	if h.n > 0 && h.n <= bufSize {
		k := copy(h.buf[h.n:], s)
		h.n += k
		if h.n < bufSize {
			return size, nil
		}
		s = s[k:]
		h.flush()
	}
	if len(s) > bufSize {
		h.initSeed()
		for len(s) > bufSize {
			h.state.s = rthashString(s[:bufSize], h.state.s)
			s = s[bufSize:]
		}
	}
	copy(h.buf[:], s)
	h.n = len(s)
	return size, nil
}

// Seed returns h's seed value.
func (h *Hash) Seed() Seed {
	h.initSeed()
	return h.seed
}

// SetSeed sets h to use seed, which must have been returned by [MakeSeed]
// or by another [Hash.Seed] method.
// Two [Hash] objects with the same seed behave identically.
// Two [Hash] objects with different seeds will very likely behave differently.
// Any bytes added to h before this call will be discarded.
func (h *Hash) SetSeed(seed Seed) {
	if seed.s == 0 {
		panic("maphash: use of uninitialized Seed")
	}
	h.seed = seed
	h.state = seed
	h.n = 0
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
	h.state.s = rthash(h.buf[:h.n], h.state.s)
	h.n = 0
}

// Sum64 returns h's current 64-bit value, which depends on
// h's seed and the sequence of bytes added to h since the
// last call to [Hash.Reset] or [Hash.SetSeed].
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
	var s uint64
	for {
		s = randUint64()
		// We use seed 0 to indicate an uninitialized seed/hash,
		// so keep trying until we get a non-zero seed.
		if s != 0 {
			break
		}
	}
	return Seed{s: s}
}

// Sum appends the hash's current 64-bit value to b.
// It exists for implementing [hash.Hash].
// For direct calls, it is more efficient to use [Hash.Sum64].
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

// Comparable returns the hash of comparable value v with the given seed
// such that Comparable(s, v1) == Comparable(s, v2) if v1 == v2.
// If v != v, then the resulting hash is randomly distributed.
func Comparable[T comparable](seed Seed, v T) uint64 {
	escapeForHash(v)
	return comparableHash(v, seed)
}

// escapeForHash forces v to be on the heap, if v contains a
// non-string pointer. We cannot hash pointers to local variables,
// as the address of the local variable might change on stack growth.
// Strings are okay as the hash depends on only the content, not
// the pointer.
//
// This is essentially
//
//	if hasNonStringPointers(T) { abi.Escape(v) }
//
// Implemented as a compiler intrinsic.
func escapeForHash[T comparable](v T) { panic("intrinsic") }

// WriteComparable adds x to the data hashed by h.
func WriteComparable[T comparable](h *Hash, x T) {
	escapeForHash(x)
	// writeComparable (not in purego mode) directly operates on h.state
	// without using h.buf. Mix in the buffer length so it won't
	// commute with a buffered write, which either changes h.n or changes
	// h.state.
	if h.n != 0 {
		writeComparable(h, h.n)
	}
	writeComparable(h, x)
}

func (h *Hash) float64(f float64) {
	if f == 0 {
		h.WriteByte(0)
		return
	}
	var buf [8]byte
	if f != f {
		byteorder.LEPutUint64(buf[:], randUint64())
		h.Write(buf[:])
		return
	}
	byteorder.LEPutUint64(buf[:], math.Float64bits(f))
	h.Write(buf[:])
}

func btoi(b bool) byte {
	if b {
		return 1
	}
	return 0
}

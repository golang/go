// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package bytes/hash provides hash functions on byte sequences. These
// hash functions are intended to be used to implement hash tables or
// other data structures that need to map arbitrary strings or byte
// sequences to a uniform distribution of integers. The hash functions
// are collision-resistant but are not cryptographically secure (use
// one of the hash functions in crypto/* if you need that).
//
// The produced hashes depend only on the sequence of bytes provided
// to the Hash object, not on the way in which they are provided. For
// example, the calls
//     h.AddString("foo")
//     h.AddBytes([]byte{'f','o','o'})
//     h.AddByte('f'); h.AddByte('o'); h.AddByte('o')
// will all have the same effect.
//
// Two Hash instances in the same process using the same seed
// behave identically.
//
// Two Hash instances with the same seed in different processes are
// not guaranteed to behave identically, even if the processes share
// the same binary.
//
// Hashes are intended to be collision-resistant, even for situations
// where an adversary controls the byte sequences being hashed.
// All bits of the Hash result are close to uniformly and
// independently distributed, so can be safely restricted to a range
// using bit masking, shifting, or modular arithmetic.
package hash

import (
	"unsafe"
)

// A Seed controls the behavior of a Hash.  Two Hash objects with the
// same seed in the same process will behave identically.  Two Hash
// objects with different seeds will very likely behave differently.
type Seed struct {
	s uint64
}

// A Hash object is used to compute the hash of a byte sequence.
type Hash struct {
	seed  Seed     // initial seed used for this hash
	state Seed     // current hash of all flushed bytes
	buf   [64]byte // unflushed byte buffer
	n     int      // number of unflushed bytes
}

// AddByte adds b to the sequence of bytes hashed by h.
func (h *Hash) AddByte(b byte) {
	if h.n == len(h.buf) {
		h.flush()
	}
	h.buf[h.n] = b
	h.n++
}

// AddBytes adds b to the sequence of bytes hashed by h.
func (h *Hash) AddBytes(b []byte) {
	for h.n+len(b) > len(h.buf) {
		k := copy(h.buf[h.n:], b)
		h.n = len(h.buf)
		b = b[k:]
		h.flush()
	}
	h.n += copy(h.buf[h.n:], b)
}

// AddString adds the bytes of s to the sequence of bytes hashed by h.
func (h *Hash) AddString(s string) {
	for h.n+len(s) > len(h.buf) {
		k := copy(h.buf[h.n:], s)
		h.n = len(h.buf)
		s = s[k:]
		h.flush()
	}
	h.n += copy(h.buf[h.n:], s)
}

// Seed returns the seed value specified in the most recent call to
// SetSeed, or the initial seed if SetSeed was never called.
func (h *Hash) Seed() Seed {
	return h.seed
}

// SetSeed sets the seed used by h. Two Hash objects with the same
// seed in the same process will behave identically.  Two Hash objects
// with different seeds will very likely behave differently.  Any
// bytes added to h previous to this call will be discarded.
func (h *Hash) SetSeed(seed Seed) {
	h.seed = seed
	h.state = seed
	h.n = 0
}

// Reset discards all bytes added to h.
// (The seed remains the same.)
func (h *Hash) Reset() {
	h.state = h.seed
	h.n = 0
}

// precondition: buffer is full.
func (h *Hash) flush() {
	if h.n != len(h.buf) {
		panic("flush of partially full buffer")
	}
	h.state.s = rthash(h.buf[:], h.state.s)
	h.n = 0
}

// Hash returns a value which depends on h's seed and the sequence of
// bytes added to h (since the last call to Reset or SetSeed).
func (h *Hash) Hash() uint64 {
	return rthash(h.buf[:h.n], h.state.s)
}

// MakeSeed returns a Seed initialized using the bits in s.
// Two seeds generated with the same s are guaranteed to be equal.
// Two seeds generated with different s are very likely to be different.
// TODO: disallow this? See Alan's comment in the issue.
func MakeSeed(s uint64) Seed {
	return Seed{s: s}
}

// New returns a new Hash object. Different hash objects allocated by
// this function will very likely have different seeds.
func New() *Hash {
	s1 := uint64(runtime_fastrand())
	s2 := uint64(runtime_fastrand())
	seed := Seed{s: s1<<32 + s2}
	return &Hash{
		seed:  seed,
		state: seed,
	}
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
	// TODO: mix lo/hi? Get 64 bits some other way?
	return uint64(hi)<<32 | uint64(lo)
}

//go:linkname runtime_memhash runtime.memhash
func runtime_memhash(p unsafe.Pointer, seed, s uintptr) uintptr

// Wrapper functions so that a bytes/hash.Hash implements
// the hash.Hash and hash.Hash64 interfaces.

func (h *Hash) Write(b []byte) (int, error) {
	h.AddBytes(b)
	return len(b), nil
}
func (h *Hash) Sum(b []byte) []byte {
	x := h.Hash()
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
func (h *Hash) Sum64() uint64 {
	return h.Hash()
}
func (h *Hash) Size() int      { return 8 }
func (h *Hash) BlockSize() int { return len(h.buf) }

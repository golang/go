// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !race

package maphash

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"strings"
	"testing"
	"unsafe"
)

// Smhasher is a torture test for hash functions.
// https://code.google.com/p/smhasher/
// This code is a port of some of the Smhasher tests to Go.

// Note: due to the long running time of these tests, they are
// currently disabled in -race mode.

var fixedSeed = MakeSeed()

// Sanity checks.
// hash should not depend on values outside key.
// hash should not depend on alignment.
func TestSmhasherSanity(t *testing.T) {
	r := rand.New(rand.NewSource(1234))
	const REP = 10
	const KEYMAX = 128
	const PAD = 16
	const OFFMAX = 16
	for k := 0; k < REP; k++ {
		for n := 0; n < KEYMAX; n++ {
			for i := 0; i < OFFMAX; i++ {
				var b [KEYMAX + OFFMAX + 2*PAD]byte
				var c [KEYMAX + OFFMAX + 2*PAD]byte
				randBytes(r, b[:])
				randBytes(r, c[:])
				copy(c[PAD+i:PAD+i+n], b[PAD:PAD+n])
				if bytesHash(b[PAD:PAD+n]) != bytesHash(c[PAD+i:PAD+i+n]) {
					t.Errorf("hash depends on bytes outside key")
				}
			}
		}
	}
}

func bytesHash(b []byte) uint64 {
	var h Hash
	h.SetSeed(fixedSeed)
	h.Write(b)
	return h.Sum64()
}
func stringHash(s string) uint64 {
	var h Hash
	h.SetSeed(fixedSeed)
	h.WriteString(s)
	return h.Sum64()
}

const hashSize = 64

func randBytes(r *rand.Rand, b []byte) {
	r.Read(b) // can't fail
}

// A hashSet measures the frequency of hash collisions.
type hashSet struct {
	m map[uint64]struct{} // set of hashes added
	n int                 // number of hashes added
}

func newHashSet() *hashSet {
	return &hashSet{make(map[uint64]struct{}), 0}
}
func (s *hashSet) add(h uint64) {
	s.m[h] = struct{}{}
	s.n++
}
func (s *hashSet) addS(x string) {
	s.add(stringHash(x))
}
func (s *hashSet) addB(x []byte) {
	s.add(bytesHash(x))
}
func (s *hashSet) addS_seed(x string, seed Seed) {
	var h Hash
	h.SetSeed(seed)
	h.WriteString(x)
	s.add(h.Sum64())
}
func (s *hashSet) check(t *testing.T) {
	const SLOP = 10.0
	collisions := s.n - len(s.m)
	pairs := int64(s.n) * int64(s.n-1) / 2
	expected := float64(pairs) / math.Pow(2.0, float64(hashSize))
	stddev := math.Sqrt(expected)
	if float64(collisions) > expected+SLOP*(3*stddev+1) {
		t.Errorf("unexpected number of collisions: got=%d mean=%f stddev=%f", collisions, expected, stddev)
	}
}

// a string plus adding zeros must make distinct hashes
func TestSmhasherAppendedZeros(t *testing.T) {
	s := "hello" + strings.Repeat("\x00", 256)
	h := newHashSet()
	for i := 0; i <= len(s); i++ {
		h.addS(s[:i])
	}
	h.check(t)
}

// All 0-3 byte strings have distinct hashes.
func TestSmhasherSmallKeys(t *testing.T) {
	h := newHashSet()
	var b [3]byte
	for i := 0; i < 256; i++ {
		b[0] = byte(i)
		h.addB(b[:1])
		for j := 0; j < 256; j++ {
			b[1] = byte(j)
			h.addB(b[:2])
			if !testing.Short() {
				for k := 0; k < 256; k++ {
					b[2] = byte(k)
					h.addB(b[:3])
				}
			}
		}
	}
	h.check(t)
}

// Different length strings of all zeros have distinct hashes.
func TestSmhasherZeros(t *testing.T) {
	N := 256 * 1024
	if testing.Short() {
		N = 1024
	}
	h := newHashSet()
	b := make([]byte, N)
	for i := 0; i <= N; i++ {
		h.addB(b[:i])
	}
	h.check(t)
}

// Strings with up to two nonzero bytes all have distinct hashes.
func TestSmhasherTwoNonzero(t *testing.T) {
	if runtime.GOARCH == "wasm" {
		t.Skip("Too slow on wasm")
	}
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}
	h := newHashSet()
	for n := 2; n <= 16; n++ {
		twoNonZero(h, n)
	}
	h.check(t)
}
func twoNonZero(h *hashSet, n int) {
	b := make([]byte, n)

	// all zero
	h.addB(b)

	// one non-zero byte
	for i := 0; i < n; i++ {
		for x := 1; x < 256; x++ {
			b[i] = byte(x)
			h.addB(b)
			b[i] = 0
		}
	}

	// two non-zero bytes
	for i := 0; i < n; i++ {
		for x := 1; x < 256; x++ {
			b[i] = byte(x)
			for j := i + 1; j < n; j++ {
				for y := 1; y < 256; y++ {
					b[j] = byte(y)
					h.addB(b)
					b[j] = 0
				}
			}
			b[i] = 0
		}
	}
}

// Test strings with repeats, like "abcdabcdabcdabcd..."
func TestSmhasherCyclic(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}
	r := rand.New(rand.NewSource(1234))
	const REPEAT = 8
	const N = 1000000
	for n := 4; n <= 12; n++ {
		h := newHashSet()
		b := make([]byte, REPEAT*n)
		for i := 0; i < N; i++ {
			b[0] = byte(i * 79 % 97)
			b[1] = byte(i * 43 % 137)
			b[2] = byte(i * 151 % 197)
			b[3] = byte(i * 199 % 251)
			randBytes(r, b[4:n])
			for j := n; j < n*REPEAT; j++ {
				b[j] = b[j-n]
			}
			h.addB(b)
		}
		h.check(t)
	}
}

// Test strings with only a few bits set
func TestSmhasherSparse(t *testing.T) {
	if runtime.GOARCH == "wasm" {
		t.Skip("Too slow on wasm")
	}
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}
	sparse(t, 32, 6)
	sparse(t, 40, 6)
	sparse(t, 48, 5)
	sparse(t, 56, 5)
	sparse(t, 64, 5)
	sparse(t, 96, 4)
	sparse(t, 256, 3)
	sparse(t, 2048, 2)
}
func sparse(t *testing.T, n int, k int) {
	b := make([]byte, n/8)
	h := newHashSet()
	setbits(h, b, 0, k)
	h.check(t)
}

// set up to k bits at index i and greater
func setbits(h *hashSet, b []byte, i int, k int) {
	h.addB(b)
	if k == 0 {
		return
	}
	for j := i; j < len(b)*8; j++ {
		b[j/8] |= byte(1 << uint(j&7))
		setbits(h, b, j+1, k-1)
		b[j/8] &= byte(^(1 << uint(j&7)))
	}
}

// Test all possible combinations of n blocks from the set s.
// "permutation" is a bad name here, but it is what Smhasher uses.
func TestSmhasherPermutation(t *testing.T) {
	if runtime.GOARCH == "wasm" {
		t.Skip("Too slow on wasm")
	}
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}
	permutation(t, []uint32{0, 1, 2, 3, 4, 5, 6, 7}, 8)
	permutation(t, []uint32{0, 1 << 29, 2 << 29, 3 << 29, 4 << 29, 5 << 29, 6 << 29, 7 << 29}, 8)
	permutation(t, []uint32{0, 1}, 20)
	permutation(t, []uint32{0, 1 << 31}, 20)
	permutation(t, []uint32{0, 1, 2, 3, 4, 5, 6, 7, 1 << 29, 2 << 29, 3 << 29, 4 << 29, 5 << 29, 6 << 29, 7 << 29}, 6)
}
func permutation(t *testing.T, s []uint32, n int) {
	b := make([]byte, n*4)
	h := newHashSet()
	genPerm(h, b, s, 0)
	h.check(t)
}
func genPerm(h *hashSet, b []byte, s []uint32, n int) {
	h.addB(b[:n])
	if n == len(b) {
		return
	}
	for _, v := range s {
		b[n] = byte(v)
		b[n+1] = byte(v >> 8)
		b[n+2] = byte(v >> 16)
		b[n+3] = byte(v >> 24)
		genPerm(h, b, s, n+4)
	}
}

type key interface {
	clear()              // set bits all to 0
	random(r *rand.Rand) // set key to something random
	bits() int           // how many bits key has
	flipBit(i int)       // flip bit i of the key
	hash() uint64        // hash the key
	name() string        // for error reporting
}

type bytesKey struct {
	b []byte
}

func (k *bytesKey) clear() {
	for i := range k.b {
		k.b[i] = 0
	}
}
func (k *bytesKey) random(r *rand.Rand) {
	randBytes(r, k.b)
}
func (k *bytesKey) bits() int {
	return len(k.b) * 8
}
func (k *bytesKey) flipBit(i int) {
	k.b[i>>3] ^= byte(1 << uint(i&7))
}
func (k *bytesKey) hash() uint64 {
	return bytesHash(k.b)
}
func (k *bytesKey) name() string {
	return fmt.Sprintf("bytes%d", len(k.b))
}

// Flipping a single bit of a key should flip each output bit with 50% probability.
func TestSmhasherAvalanche(t *testing.T) {
	if runtime.GOARCH == "wasm" {
		t.Skip("Too slow on wasm")
	}
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}
	avalancheTest1(t, &bytesKey{make([]byte, 2)})
	avalancheTest1(t, &bytesKey{make([]byte, 4)})
	avalancheTest1(t, &bytesKey{make([]byte, 8)})
	avalancheTest1(t, &bytesKey{make([]byte, 16)})
	avalancheTest1(t, &bytesKey{make([]byte, 32)})
	avalancheTest1(t, &bytesKey{make([]byte, 200)})
}
func avalancheTest1(t *testing.T, k key) {
	const REP = 100000
	r := rand.New(rand.NewSource(1234))
	n := k.bits()

	// grid[i][j] is a count of whether flipping
	// input bit i affects output bit j.
	grid := make([][hashSize]int, n)

	for z := 0; z < REP; z++ {
		// pick a random key, hash it
		k.random(r)
		h := k.hash()

		// flip each bit, hash & compare the results
		for i := 0; i < n; i++ {
			k.flipBit(i)
			d := h ^ k.hash()
			k.flipBit(i)

			// record the effects of that bit flip
			g := &grid[i]
			for j := 0; j < hashSize; j++ {
				g[j] += int(d & 1)
				d >>= 1
			}
		}
	}

	// Each entry in the grid should be about REP/2.
	// More precisely, we did N = k.bits() * hashSize experiments where
	// each is the sum of REP coin flips. We want to find bounds on the
	// sum of coin flips such that a truly random experiment would have
	// all sums inside those bounds with 99% probability.
	N := n * hashSize
	var c float64
	// find c such that Prob(mean-c*stddev < x < mean+c*stddev)^N > .9999
	for c = 0.0; math.Pow(math.Erf(c/math.Sqrt(2)), float64(N)) < .9999; c += .1 {
	}
	c *= 11.0 // allowed slack: 40% to 60% - we don't need to be perfectly random
	mean := .5 * REP
	stddev := .5 * math.Sqrt(REP)
	low := int(mean - c*stddev)
	high := int(mean + c*stddev)
	for i := 0; i < n; i++ {
		for j := 0; j < hashSize; j++ {
			x := grid[i][j]
			if x < low || x > high {
				t.Errorf("bad bias for %s bit %d -> bit %d: %d/%d\n", k.name(), i, j, x, REP)
			}
		}
	}
}

// All bit rotations of a set of distinct keys
func TestSmhasherWindowed(t *testing.T) {
	windowed(t, &bytesKey{make([]byte, 128)})
}
func windowed(t *testing.T, k key) {
	if runtime.GOARCH == "wasm" {
		t.Skip("Too slow on wasm")
	}
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}
	const BITS = 16

	for r := 0; r < k.bits(); r++ {
		h := newHashSet()
		for i := 0; i < 1<<BITS; i++ {
			k.clear()
			for j := 0; j < BITS; j++ {
				if i>>uint(j)&1 != 0 {
					k.flipBit((j + r) % k.bits())
				}
			}
			h.add(k.hash())
		}
		h.check(t)
	}
}

// All keys of the form prefix + [A-Za-z0-9]*N + suffix.
func TestSmhasherText(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}
	text(t, "Foo", "Bar")
	text(t, "FooBar", "")
	text(t, "", "FooBar")
}
func text(t *testing.T, prefix, suffix string) {
	const N = 4
	const S = "ABCDEFGHIJKLMNOPQRSTabcdefghijklmnopqrst0123456789"
	const L = len(S)
	b := make([]byte, len(prefix)+N+len(suffix))
	copy(b, prefix)
	copy(b[len(prefix)+N:], suffix)
	h := newHashSet()
	c := b[len(prefix):]
	for i := 0; i < L; i++ {
		c[0] = S[i]
		for j := 0; j < L; j++ {
			c[1] = S[j]
			for k := 0; k < L; k++ {
				c[2] = S[k]
				for x := 0; x < L; x++ {
					c[3] = S[x]
					h.addB(b)
				}
			}
		}
	}
	h.check(t)
}

// Make sure different seed values generate different hashes.
func TestSmhasherSeed(t *testing.T) {
	if unsafe.Sizeof(uintptr(0)) == 4 {
		t.Skip("32-bit platforms don't have ideal seed-input distributions (see issue 33988)")
	}
	h := newHashSet()
	const N = 100000
	s := "hello"
	for i := 0; i < N; i++ {
		h.addS_seed(s, Seed{s: uint64(i + 1)})
		h.addS_seed(s, Seed{s: uint64(i+1) << 32}) // make sure high bits are used
	}
	h.check(t)
}

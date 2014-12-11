// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"fmt"
	"math"
	"math/rand"
	. "runtime"
	"strings"
	"testing"
)

// Smhasher is a torture test for hash functions.
// https://code.google.com/p/smhasher/
// This code is a port of some of the Smhasher tests to Go.
//
// The current AES hash function passes Smhasher.  Our fallback
// hash functions don't, so we only enable the difficult tests when
// we know the AES implementation is available.

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
				if BytesHash(b[PAD:PAD+n], 0) != BytesHash(c[PAD+i:PAD+i+n], 0) {
					t.Errorf("hash depends on bytes outside key")
				}
			}
		}
	}
}

type HashSet struct {
	m map[uintptr]struct{} // set of hashes added
	n int                  // number of hashes added
}

func newHashSet() *HashSet {
	return &HashSet{make(map[uintptr]struct{}), 0}
}
func (s *HashSet) add(h uintptr) {
	s.m[h] = struct{}{}
	s.n++
}
func (s *HashSet) addS(x string) {
	s.add(StringHash(x, 0))
}
func (s *HashSet) addB(x []byte) {
	s.add(BytesHash(x, 0))
}
func (s *HashSet) addS_seed(x string, seed uintptr) {
	s.add(StringHash(x, seed))
}
func (s *HashSet) check(t *testing.T) {
	const SLOP = 10.0
	collisions := s.n - len(s.m)
	//fmt.Printf("%d/%d\n", len(s.m), s.n)
	pairs := int64(s.n) * int64(s.n-1) / 2
	expected := float64(pairs) / math.Pow(2.0, float64(hashSize))
	stddev := math.Sqrt(expected)
	if float64(collisions) > expected+SLOP*3*stddev {
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
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}
	h := newHashSet()
	for n := 2; n <= 16; n++ {
		twoNonZero(h, n)
	}
	h.check(t)
}
func twoNonZero(h *HashSet, n int) {
	b := make([]byte, n)

	// all zero
	h.addB(b[:])

	// one non-zero byte
	for i := 0; i < n; i++ {
		for x := 1; x < 256; x++ {
			b[i] = byte(x)
			h.addB(b[:])
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
					h.addB(b[:])
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
func setbits(h *HashSet, b []byte, i int, k int) {
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
func genPerm(h *HashSet, b []byte, s []uint32, n int) {
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

type Key interface {
	clear()              // set bits all to 0
	random(r *rand.Rand) // set key to something random
	bits() int           // how many bits key has
	flipBit(i int)       // flip bit i of the key
	hash() uintptr       // hash the key
	name() string        // for error reporting
}

type BytesKey struct {
	b []byte
}

func (k *BytesKey) clear() {
	for i := range k.b {
		k.b[i] = 0
	}
}
func (k *BytesKey) random(r *rand.Rand) {
	randBytes(r, k.b)
}
func (k *BytesKey) bits() int {
	return len(k.b) * 8
}
func (k *BytesKey) flipBit(i int) {
	k.b[i>>3] ^= byte(1 << uint(i&7))
}
func (k *BytesKey) hash() uintptr {
	return BytesHash(k.b, 0)
}
func (k *BytesKey) name() string {
	return fmt.Sprintf("bytes%d", len(k.b))
}

type Int32Key struct {
	i uint32
}

func (k *Int32Key) clear() {
	k.i = 0
}
func (k *Int32Key) random(r *rand.Rand) {
	k.i = r.Uint32()
}
func (k *Int32Key) bits() int {
	return 32
}
func (k *Int32Key) flipBit(i int) {
	k.i ^= 1 << uint(i)
}
func (k *Int32Key) hash() uintptr {
	return Int32Hash(k.i, 0)
}
func (k *Int32Key) name() string {
	return "int32"
}

type Int64Key struct {
	i uint64
}

func (k *Int64Key) clear() {
	k.i = 0
}
func (k *Int64Key) random(r *rand.Rand) {
	k.i = uint64(r.Uint32()) + uint64(r.Uint32())<<32
}
func (k *Int64Key) bits() int {
	return 64
}
func (k *Int64Key) flipBit(i int) {
	k.i ^= 1 << uint(i)
}
func (k *Int64Key) hash() uintptr {
	return Int64Hash(k.i, 0)
}
func (k *Int64Key) name() string {
	return "int64"
}

type EfaceKey struct {
	i interface{}
}

func (k *EfaceKey) clear() {
	k.i = nil
}
func (k *EfaceKey) random(r *rand.Rand) {
	k.i = uint64(r.Int63())
}
func (k *EfaceKey) bits() int {
	// use 64 bits.  This tests inlined interfaces
	// on 64-bit targets and indirect interfaces on
	// 32-bit targets.
	return 64
}
func (k *EfaceKey) flipBit(i int) {
	k.i = k.i.(uint64) ^ uint64(1)<<uint(i)
}
func (k *EfaceKey) hash() uintptr {
	return EfaceHash(k.i, 0)
}
func (k *EfaceKey) name() string {
	return "Eface"
}

type IfaceKey struct {
	i interface {
		F()
	}
}
type fInter uint64

func (x fInter) F() {
}

func (k *IfaceKey) clear() {
	k.i = nil
}
func (k *IfaceKey) random(r *rand.Rand) {
	k.i = fInter(r.Int63())
}
func (k *IfaceKey) bits() int {
	// use 64 bits.  This tests inlined interfaces
	// on 64-bit targets and indirect interfaces on
	// 32-bit targets.
	return 64
}
func (k *IfaceKey) flipBit(i int) {
	k.i = k.i.(fInter) ^ fInter(1)<<uint(i)
}
func (k *IfaceKey) hash() uintptr {
	return IfaceHash(k.i, 0)
}
func (k *IfaceKey) name() string {
	return "Iface"
}

// Flipping a single bit of a key should flip each output bit with 50% probability.
func TestSmhasherAvalanche(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}
	avalancheTest1(t, &BytesKey{make([]byte, 2)})
	avalancheTest1(t, &BytesKey{make([]byte, 4)})
	avalancheTest1(t, &BytesKey{make([]byte, 8)})
	avalancheTest1(t, &BytesKey{make([]byte, 16)})
	avalancheTest1(t, &BytesKey{make([]byte, 32)})
	avalancheTest1(t, &BytesKey{make([]byte, 200)})
	avalancheTest1(t, &Int32Key{})
	avalancheTest1(t, &Int64Key{})
	avalancheTest1(t, &EfaceKey{})
	avalancheTest1(t, &IfaceKey{})
}
func avalancheTest1(t *testing.T, k Key) {
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
	// each is the sum of REP coin flips.  We want to find bounds on the
	// sum of coin flips such that a truly random experiment would have
	// all sums inside those bounds with 99% probability.
	N := n * hashSize
	var c float64
	// find c such that Prob(mean-c*stddev < x < mean+c*stddev)^N > .9999
	for c = 0.0; math.Pow(math.Erf(c/math.Sqrt(2)), float64(N)) < .9999; c += .1 {
	}
	c *= 4.0 // allowed slack - we don't need to be perfectly random
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
	windowed(t, &Int32Key{})
	windowed(t, &Int64Key{})
	windowed(t, &BytesKey{make([]byte, 128)})
}
func windowed(t *testing.T, k Key) {
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
	h := newHashSet()
	const N = 100000
	s := "hello"
	for i := 0; i < N; i++ {
		h.addS_seed(s, uintptr(i))
	}
	h.check(t)
}

// size of the hash output (32 or 64 bits)
const hashSize = 32 + int(^uintptr(0)>>63<<5)

func randBytes(r *rand.Rand, b []byte) {
	for i := range b {
		b[i] = byte(r.Uint32())
	}
}

func benchmarkHash(b *testing.B, n int) {
	s := strings.Repeat("A", n)

	for i := 0; i < b.N; i++ {
		StringHash(s, 0)
	}
	b.SetBytes(int64(n))
}

func BenchmarkHash5(b *testing.B)     { benchmarkHash(b, 5) }
func BenchmarkHash16(b *testing.B)    { benchmarkHash(b, 16) }
func BenchmarkHash64(b *testing.B)    { benchmarkHash(b, 64) }
func BenchmarkHash1024(b *testing.B)  { benchmarkHash(b, 1024) }
func BenchmarkHash65536(b *testing.B) { benchmarkHash(b, 65536) }

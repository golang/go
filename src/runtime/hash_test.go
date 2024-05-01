// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"encoding/binary"
	"fmt"
	"internal/race"
	"internal/testenv"
	"math"
	"math/rand"
	"os"
	. "runtime"
	"slices"
	"strings"
	"testing"
	"unsafe"
)

func TestMemHash32Equality(t *testing.T) {
	if *UseAeshash {
		t.Skip("skipping since AES hash implementation is used")
	}
	var b [4]byte
	r := rand.New(rand.NewSource(1234))
	seed := uintptr(r.Uint64())
	for i := 0; i < 100; i++ {
		randBytes(r, b[:])
		got := MemHash32(unsafe.Pointer(&b), seed)
		want := MemHash(unsafe.Pointer(&b), seed, 4)
		if got != want {
			t.Errorf("MemHash32(%x, %v) = %v; want %v", b, seed, got, want)
		}
	}
}

func TestMemHash64Equality(t *testing.T) {
	if *UseAeshash {
		t.Skip("skipping since AES hash implementation is used")
	}
	var b [8]byte
	r := rand.New(rand.NewSource(1234))
	seed := uintptr(r.Uint64())
	for i := 0; i < 100; i++ {
		randBytes(r, b[:])
		got := MemHash64(unsafe.Pointer(&b), seed)
		want := MemHash(unsafe.Pointer(&b), seed, 8)
		if got != want {
			t.Errorf("MemHash64(%x, %v) = %v; want %v", b, seed, got, want)
		}
	}
}

// Smhasher is a torture test for hash functions.
// https://code.google.com/p/smhasher/
// This code is a port of some of the Smhasher tests to Go.
//
// The current AES hash function passes Smhasher. Our fallback
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
	list []uintptr // list of hashes added
}

func newHashSet() *HashSet {
	return &HashSet{list: make([]uintptr, 0, 1024)}
}
func (s *HashSet) add(h uintptr) {
	s.list = append(s.list, h)
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
	list := s.list
	slices.Sort(list)

	collisions := 0
	for i := 1; i < len(list); i++ {
		if list[i] == list[i-1] {
			collisions++
		}
	}
	n := len(list)

	const SLOP = 50.0
	pairs := int64(n) * int64(n-1) / 2
	expected := float64(pairs) / math.Pow(2.0, float64(hashSize))
	stddev := math.Sqrt(expected)
	if float64(collisions) > expected+SLOP*(3*stddev+1) {
		t.Errorf("unexpected number of collisions: got=%d mean=%f stddev=%f threshold=%f", collisions, expected, stddev, expected+SLOP*(3*stddev+1))
	}
	// Reset for reuse
	s.list = s.list[:0]
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
	if race.Enabled {
		t.Skip("Too long for race mode")
	}
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
	if GOARCH == "wasm" {
		t.Skip("Too slow on wasm")
	}
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}
	if race.Enabled {
		t.Skip("Too long for race mode")
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
	if race.Enabled {
		t.Skip("Too long for race mode")
	}
	r := rand.New(rand.NewSource(1234))
	const REPEAT = 8
	const N = 1000000
	h := newHashSet()
	for n := 4; n <= 12; n++ {
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
	if GOARCH == "wasm" {
		t.Skip("Too slow on wasm")
	}
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}
	h := newHashSet()
	sparse(t, h, 32, 6)
	sparse(t, h, 40, 6)
	sparse(t, h, 48, 5)
	sparse(t, h, 56, 5)
	sparse(t, h, 64, 5)
	sparse(t, h, 96, 4)
	sparse(t, h, 256, 3)
	sparse(t, h, 2048, 2)
}
func sparse(t *testing.T, h *HashSet, n int, k int) {
	b := make([]byte, n/8)
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
	if GOARCH == "wasm" {
		t.Skip("Too slow on wasm")
	}
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}
	if race.Enabled {
		t.Skip("Too long for race mode")
	}
	h := newHashSet()
	permutation(t, h, []uint32{0, 1, 2, 3, 4, 5, 6, 7}, 8)
	permutation(t, h, []uint32{0, 1 << 29, 2 << 29, 3 << 29, 4 << 29, 5 << 29, 6 << 29, 7 << 29}, 8)
	permutation(t, h, []uint32{0, 1}, 20)
	permutation(t, h, []uint32{0, 1 << 31}, 20)
	permutation(t, h, []uint32{0, 1, 2, 3, 4, 5, 6, 7, 1 << 29, 2 << 29, 3 << 29, 4 << 29, 5 << 29, 6 << 29, 7 << 29}, 6)
}
func permutation(t *testing.T, h *HashSet, s []uint32, n int) {
	b := make([]byte, n*4)
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
	clear(k.b)
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
	i any
}

func (k *EfaceKey) clear() {
	k.i = nil
}
func (k *EfaceKey) random(r *rand.Rand) {
	k.i = uint64(r.Int63())
}
func (k *EfaceKey) bits() int {
	// use 64 bits. This tests inlined interfaces
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
	// use 64 bits. This tests inlined interfaces
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
	if GOARCH == "wasm" {
		t.Skip("Too slow on wasm")
	}
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}
	if race.Enabled {
		t.Skip("Too long for race mode")
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
	if race.Enabled {
		t.Skip("Too long for race mode")
	}
	h := newHashSet()
	t.Logf("32 bit keys")
	windowed(t, h, &Int32Key{})
	t.Logf("64 bit keys")
	windowed(t, h, &Int64Key{})
	t.Logf("string keys")
	windowed(t, h, &BytesKey{make([]byte, 128)})
}
func windowed(t *testing.T, h *HashSet, k Key) {
	if GOARCH == "wasm" {
		t.Skip("Too slow on wasm")
	}
	if PtrSize == 4 {
		// This test tends to be flaky on 32-bit systems.
		// There's not enough bits in the hash output, so we
		// expect a nontrivial number of collisions, and it is
		// often quite a bit higher than expected. See issue 43130.
		t.Skip("Flaky on 32-bit systems")
	}
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}
	const BITS = 16

	for r := 0; r < k.bits(); r++ {
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
	h := newHashSet()
	text(t, h, "Foo", "Bar")
	text(t, h, "FooBar", "")
	text(t, h, "", "FooBar")
}
func text(t *testing.T, h *HashSet, prefix, suffix string) {
	const N = 4
	const S = "ABCDEFGHIJKLMNOPQRSTabcdefghijklmnopqrst0123456789"
	const L = len(S)
	b := make([]byte, len(prefix)+N+len(suffix))
	copy(b, prefix)
	copy(b[len(prefix)+N:], suffix)
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

func TestIssue66841(t *testing.T) {
	testenv.MustHaveExec(t)
	if *UseAeshash && os.Getenv("TEST_ISSUE_66841") == "" {
		// We want to test the backup hash, so if we're running on a machine
		// that uses aeshash, exec ourselves while turning aes off.
		cmd := testenv.CleanCmdEnv(testenv.Command(t, os.Args[0], "-test.run=^TestIssue66841$"))
		cmd.Env = append(cmd.Env, "GODEBUG=cpu.aes=off", "TEST_ISSUE_66841=1")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Errorf("%s", string(out))
		}
		// Fall through. Might as well run this test when aeshash is on also.
	}
	h := newHashSet()
	var b [16]byte
	binary.LittleEndian.PutUint64(b[:8], 0xe7037ed1a0b428db) // runtime.m2
	for i := 0; i < 1000; i++ {
		binary.LittleEndian.PutUint64(b[8:], uint64(i))
		h.addB(b[:])
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

func TestArrayHash(t *testing.T) {
	// Make sure that "" in arrays hash correctly. The hash
	// should at least scramble the input seed so that, e.g.,
	// {"","foo"} and {"foo",""} have different hashes.

	// If the hash is bad, then all (8 choose 4) = 70 keys
	// have the same hash. If so, we allocate 70/8 = 8
	// overflow buckets. If the hash is good we don't
	// normally allocate any overflow buckets, and the
	// probability of even one or two overflows goes down rapidly.
	// (There is always 1 allocation of the bucket array. The map
	// header is allocated on the stack.)
	f := func() {
		// Make the key type at most 128 bytes. Otherwise,
		// we get an allocation per key.
		type key [8]string
		m := make(map[key]bool, 70)

		// fill m with keys that have 4 "foo"s and 4 ""s.
		for i := 0; i < 256; i++ {
			var k key
			cnt := 0
			for j := uint(0); j < 8; j++ {
				if i>>j&1 != 0 {
					k[j] = "foo"
					cnt++
				}
			}
			if cnt == 4 {
				m[k] = true
			}
		}
		if len(m) != 70 {
			t.Errorf("bad test: (8 choose 4) should be 70, not %d", len(m))
		}
	}
	if n := testing.AllocsPerRun(10, f); n > 6 {
		t.Errorf("too many allocs %f - hash not balanced", n)
	}
}
func TestStructHash(t *testing.T) {
	// See the comment in TestArrayHash.
	f := func() {
		type key struct {
			a, b, c, d, e, f, g, h string
		}
		m := make(map[key]bool, 70)

		// fill m with keys that have 4 "foo"s and 4 ""s.
		for i := 0; i < 256; i++ {
			var k key
			cnt := 0
			if i&1 != 0 {
				k.a = "foo"
				cnt++
			}
			if i&2 != 0 {
				k.b = "foo"
				cnt++
			}
			if i&4 != 0 {
				k.c = "foo"
				cnt++
			}
			if i&8 != 0 {
				k.d = "foo"
				cnt++
			}
			if i&16 != 0 {
				k.e = "foo"
				cnt++
			}
			if i&32 != 0 {
				k.f = "foo"
				cnt++
			}
			if i&64 != 0 {
				k.g = "foo"
				cnt++
			}
			if i&128 != 0 {
				k.h = "foo"
				cnt++
			}
			if cnt == 4 {
				m[k] = true
			}
		}
		if len(m) != 70 {
			t.Errorf("bad test: (8 choose 4) should be 70, not %d", len(m))
		}
	}
	if n := testing.AllocsPerRun(10, f); n > 6 {
		t.Errorf("too many allocs %f - hash not balanced", n)
	}
}

var sink uint64

func BenchmarkAlignedLoad(b *testing.B) {
	var buf [16]byte
	p := unsafe.Pointer(&buf[0])
	var s uint64
	for i := 0; i < b.N; i++ {
		s += ReadUnaligned64(p)
	}
	sink = s
}

func BenchmarkUnalignedLoad(b *testing.B) {
	var buf [16]byte
	p := unsafe.Pointer(&buf[1])
	var s uint64
	for i := 0; i < b.N; i++ {
		s += ReadUnaligned64(p)
	}
	sink = s
}

func TestCollisions(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}
	for i := 0; i < 16; i++ {
		for j := 0; j < 16; j++ {
			if j == i {
				continue
			}
			var a [16]byte
			m := make(map[uint16]struct{}, 1<<16)
			for n := 0; n < 1<<16; n++ {
				a[i] = byte(n)
				a[j] = byte(n >> 8)
				m[uint16(BytesHash(a[:], 0))] = struct{}{}
			}
			// N balls in N bins, for N=65536
			avg := 41427
			stdDev := 123
			if len(m) < avg-40*stdDev || len(m) > avg+40*stdDev {
				t.Errorf("bad number of collisions i=%d j=%d outputs=%d out of 65536\n", i, j, len(m))
			}
		}
	}
}

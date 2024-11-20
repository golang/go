// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fipstest_test

// TODO(fips, #69982): move to the crypto/sha3 package once it exists.

import (
	"bytes"
	"crypto/internal/cryptotest"
	"crypto/internal/fips"
	. "crypto/internal/fips/sha3"
	"encoding"
	"encoding/hex"
	"fmt"
	"io"
	"math/rand"
	"strings"
	"testing"
)

// Sum224 returns the SHA3-224 digest of the data.
func Sum224(data []byte) (digest [28]byte) {
	h := New224()
	h.Write(data)
	h.Sum(digest[:0])
	return
}

// Sum256 returns the SHA3-256 digest of the data.
func Sum256(data []byte) (digest [32]byte) {
	h := New256()
	h.Write(data)
	h.Sum(digest[:0])
	return
}

// Sum384 returns the SHA3-384 digest of the data.
func Sum384(data []byte) (digest [48]byte) {
	h := New384()
	h.Write(data)
	h.Sum(digest[:0])
	return
}

// Sum512 returns the SHA3-512 digest of the data.
func Sum512(data []byte) (digest [64]byte) {
	h := New512()
	h.Write(data)
	h.Sum(digest[:0])
	return
}

// ShakeSum128 writes an arbitrary-length digest of data into hash.
func ShakeSum128(hash, data []byte) {
	h := NewShake128()
	h.Write(data)
	h.Read(hash)
}

// ShakeSum256 writes an arbitrary-length digest of data into hash.
func ShakeSum256(hash, data []byte) {
	h := NewShake256()
	h.Write(data)
	h.Read(hash)
}

const testString = "brekeccakkeccak koax koax"

// testDigests contains functions returning hash.Hash instances
// with output-length equal to the KAT length for SHA-3, Keccak
// and SHAKE instances.
var testDigests = map[string]func() *Digest{
	"SHA3-224":   New224,
	"SHA3-256":   New256,
	"SHA3-384":   New384,
	"SHA3-512":   New512,
	"Keccak-256": NewLegacyKeccak256,
	"Keccak-512": NewLegacyKeccak512,
}

// testShakes contains functions that return *sha3.SHAKE instances for
// with output-length equal to the KAT length.
var testShakes = map[string]struct {
	constructor  func(N []byte, S []byte) *SHAKE
	defAlgoName  string
	defCustomStr string
}{
	// NewCShake without customization produces same result as SHAKE
	"SHAKE128":  {NewCShake128, "", ""},
	"SHAKE256":  {NewCShake256, "", ""},
	"cSHAKE128": {NewCShake128, "CSHAKE128", "CustomString"},
	"cSHAKE256": {NewCShake256, "CSHAKE256", "CustomString"},
}

// decodeHex converts a hex-encoded string into a raw byte string.
func decodeHex(s string) []byte {
	b, err := hex.DecodeString(s)
	if err != nil {
		panic(err)
	}
	return b
}

// TestKeccak does a basic test of the non-standardized Keccak hash functions.
func TestKeccak(t *testing.T) {
	cryptotest.TestAllImplementations(t, "sha3", testKeccak)
}

func testKeccak(t *testing.T) {
	tests := []struct {
		fn   func() *Digest
		data []byte
		want string
	}{
		{
			NewLegacyKeccak256,
			[]byte("abc"),
			"4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45",
		},
		{
			NewLegacyKeccak512,
			[]byte("abc"),
			"18587dc2ea106b9a1563e32b3312421ca164c7f1f07bc922a9c83d77cea3a1e5d0c69910739025372dc14ac9642629379540c17e2a65b19d77aa511a9d00bb96",
		},
	}

	for _, u := range tests {
		h := u.fn()
		h.Write(u.data)
		got := h.Sum(nil)
		want := decodeHex(u.want)
		if !bytes.Equal(got, want) {
			t.Errorf("unexpected hash for size %d: got '%x' want '%s'", h.Size()*8, got, u.want)
		}
	}
}

// TestShakeSum tests that the output of Sum matches the output of Read.
func TestShakeSum(t *testing.T) {
	cryptotest.TestAllImplementations(t, "sha3", testShakeSum)
}

func testShakeSum(t *testing.T) {
	tests := [...]struct {
		name        string
		hash        *SHAKE
		expectedLen int
	}{
		{"SHAKE128", NewShake128(), 32},
		{"SHAKE256", NewShake256(), 64},
		{"cSHAKE128", NewCShake128([]byte{'X'}, nil), 32},
		{"cSHAKE256", NewCShake256([]byte{'X'}, nil), 64},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			s := test.hash.Sum(nil)
			if len(s) != test.expectedLen {
				t.Errorf("Unexpected digest length: got %d, want %d", len(s), test.expectedLen)
			}
			r := make([]byte, test.expectedLen)
			test.hash.Read(r)
			if !bytes.Equal(s, r) {
				t.Errorf("Mismatch between Sum and Read:\nSum:  %s\nRead: %s", hex.EncodeToString(s), hex.EncodeToString(r))
			}
		})
	}
}

// TestUnalignedWrite tests that writing data in an arbitrary pattern with
// small input buffers.
func TestUnalignedWrite(t *testing.T) {
	cryptotest.TestAllImplementations(t, "sha3", testUnalignedWrite)
}

func testUnalignedWrite(t *testing.T) {
	buf := sequentialBytes(0x10000)
	for alg, df := range testDigests {
		d := df()
		d.Reset()
		d.Write(buf)
		want := d.Sum(nil)
		d.Reset()
		for i := 0; i < len(buf); {
			// Cycle through offsets which make a 137 byte sequence.
			// Because 137 is prime this sequence should exercise all corner cases.
			offsets := [17]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1}
			for _, j := range offsets {
				if v := len(buf) - i; v < j {
					j = v
				}
				d.Write(buf[i : i+j])
				i += j
			}
		}
		got := d.Sum(nil)
		if !bytes.Equal(got, want) {
			t.Errorf("Unaligned writes, alg=%s\ngot %q, want %q", alg, got, want)
		}
	}

	// Same for SHAKE
	for alg, df := range testShakes {
		want := make([]byte, 16)
		got := make([]byte, 16)
		d := df.constructor([]byte(df.defAlgoName), []byte(df.defCustomStr))

		d.Reset()
		d.Write(buf)
		d.Read(want)
		d.Reset()
		for i := 0; i < len(buf); {
			// Cycle through offsets which make a 137 byte sequence.
			// Because 137 is prime this sequence should exercise all corner cases.
			offsets := [17]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1}
			for _, j := range offsets {
				if v := len(buf) - i; v < j {
					j = v
				}
				d.Write(buf[i : i+j])
				i += j
			}
		}
		d.Read(got)
		if !bytes.Equal(got, want) {
			t.Errorf("Unaligned writes, alg=%s\ngot %q, want %q", alg, got, want)
		}
	}
}

// TestAppend checks that appending works when reallocation is necessary.
func TestAppend(t *testing.T) {
	cryptotest.TestAllImplementations(t, "sha3", testAppend)
}

func testAppend(t *testing.T) {
	d := New224()

	for capacity := 2; capacity <= 66; capacity += 64 {
		// The first time around the loop, Sum will have to reallocate.
		// The second time, it will not.
		buf := make([]byte, 2, capacity)
		d.Reset()
		d.Write([]byte{0xcc})
		buf = d.Sum(buf)
		expected := "0000DF70ADC49B2E76EEE3A6931B93FA41841C3AF2CDF5B32A18B5478C39"
		if got := strings.ToUpper(hex.EncodeToString(buf)); got != expected {
			t.Errorf("got %s, want %s", got, expected)
		}
	}
}

// TestAppendNoRealloc tests that appending works when no reallocation is necessary.
func TestAppendNoRealloc(t *testing.T) {
	cryptotest.TestAllImplementations(t, "sha3", testAppendNoRealloc)
}

func testAppendNoRealloc(t *testing.T) {
	buf := make([]byte, 1, 200)
	d := New224()
	d.Write([]byte{0xcc})
	buf = d.Sum(buf)
	expected := "00DF70ADC49B2E76EEE3A6931B93FA41841C3AF2CDF5B32A18B5478C39"
	if got := strings.ToUpper(hex.EncodeToString(buf)); got != expected {
		t.Errorf("got %s, want %s", got, expected)
	}
}

// TestSqueezing checks that squeezing the full output a single time produces
// the same output as repeatedly squeezing the instance.
func TestSqueezing(t *testing.T) {
	cryptotest.TestAllImplementations(t, "sha3", testSqueezing)
}

func testSqueezing(t *testing.T) {
	for algo, v := range testShakes {
		d0 := v.constructor([]byte(v.defAlgoName), []byte(v.defCustomStr))
		d0.Write([]byte(testString))
		ref := make([]byte, 32)
		d0.Read(ref)

		d1 := v.constructor([]byte(v.defAlgoName), []byte(v.defCustomStr))
		d1.Write([]byte(testString))
		var multiple []byte
		for range ref {
			d1.Read(make([]byte, 0))
			one := make([]byte, 1)
			d1.Read(one)
			multiple = append(multiple, one...)
		}
		if !bytes.Equal(ref, multiple) {
			t.Errorf("%s: squeezing %d bytes one at a time failed", algo, len(ref))
		}
	}
}

// sequentialBytes produces a buffer of size consecutive bytes 0x00, 0x01, ..., used for testing.
//
// The alignment of each slice is intentionally randomized to detect alignment
// issues in the implementation. See https://golang.org/issue/37644.
// Ideally, the compiler should fuzz the alignment itself.
// (See https://golang.org/issue/35128.)
func sequentialBytes(size int) []byte {
	alignmentOffset := rand.Intn(8)
	result := make([]byte, size+alignmentOffset)[alignmentOffset:]
	for i := range result {
		result[i] = byte(i)
	}
	return result
}

func TestReset(t *testing.T) {
	cryptotest.TestAllImplementations(t, "sha3", testReset)
}

func testReset(t *testing.T) {
	out1 := make([]byte, 32)
	out2 := make([]byte, 32)

	for _, v := range testShakes {
		// Calculate hash for the first time
		c := v.constructor(nil, []byte{0x99, 0x98})
		c.Write(sequentialBytes(0x100))
		c.Read(out1)

		// Calculate hash again
		c.Reset()
		c.Write(sequentialBytes(0x100))
		c.Read(out2)

		if !bytes.Equal(out1, out2) {
			t.Error("\nExpected:\n", out1, "\ngot:\n", out2)
		}
	}
}

func TestClone(t *testing.T) {
	cryptotest.TestAllImplementations(t, "sha3", testClone)
}

func testClone(t *testing.T) {
	out1 := make([]byte, 16)
	out2 := make([]byte, 16)

	// Test for sizes smaller and larger than block size.
	for _, size := range []int{0x1, 0x100} {
		in := sequentialBytes(size)
		for _, v := range testShakes {
			h1 := v.constructor(nil, []byte{0x01})
			h1.Write([]byte{0x01})

			h2 := h1.Clone()

			h1.Write(in)
			h1.Read(out1)

			h2.Write(in)
			h2.Read(out2)

			if !bytes.Equal(out1, out2) {
				t.Error("\nExpected:\n", hex.EncodeToString(out1), "\ngot:\n", hex.EncodeToString(out2))
			}
		}
	}
}

var sinkSHA3 byte

func TestAllocations(t *testing.T) {
	cryptotest.SkipTestAllocations(t)
	t.Run("New", func(t *testing.T) {
		if allocs := testing.AllocsPerRun(10, func() {
			h := New256()
			b := []byte("ABC")
			h.Write(b)
			out := make([]byte, 0, 32)
			out = h.Sum(out)
			sinkSHA3 ^= out[0]
		}); allocs > 0 {
			t.Errorf("expected zero allocations, got %0.1f", allocs)
		}
	})
	t.Run("NewShake", func(t *testing.T) {
		if allocs := testing.AllocsPerRun(10, func() {
			h := NewShake128()
			b := []byte("ABC")
			h.Write(b)
			out := make([]byte, 0, 32)
			out = h.Sum(out)
			sinkSHA3 ^= out[0]
			h.Read(out)
			sinkSHA3 ^= out[0]
		}); allocs > 0 {
			t.Errorf("expected zero allocations, got %0.1f", allocs)
		}
	})
	t.Run("Sum", func(t *testing.T) {
		if allocs := testing.AllocsPerRun(10, func() {
			b := []byte("ABC")
			out := Sum256(b)
			sinkSHA3 ^= out[0]
		}); allocs > 0 {
			t.Errorf("expected zero allocations, got %0.1f", allocs)
		}
	})
}

func TestCSHAKEAccumulated(t *testing.T) {
	// Generated with pycryptodome@3.20.0
	//
	//    from Crypto.Hash import cSHAKE128
	//    rng = cSHAKE128.new()
	//    acc = cSHAKE128.new()
	//    for n in range(200):
	//        N = rng.read(n)
	//        for s in range(200):
	//            S = rng.read(s)
	//            c = cSHAKE128.cSHAKE_XOF(data=None, custom=S, capacity=256, function=N)
	//            c.update(rng.read(100))
	//            acc.update(c.read(200))
	//            c = cSHAKE128.cSHAKE_XOF(data=None, custom=S, capacity=256, function=N)
	//            c.update(rng.read(168))
	//            acc.update(c.read(200))
	//            c = cSHAKE128.cSHAKE_XOF(data=None, custom=S, capacity=256, function=N)
	//            c.update(rng.read(200))
	//            acc.update(c.read(200))
	//    print(acc.read(32).hex())
	//
	// and with @noble/hashes@v1.5.0
	//
	//    import { bytesToHex } from "@noble/hashes/utils";
	//    import { cshake128 } from "@noble/hashes/sha3-addons";
	//    const rng = cshake128.create();
	//    const acc = cshake128.create();
	//    for (let n = 0; n < 200; n++) {
	//        const N = rng.xof(n);
	//        for (let s = 0; s < 200; s++) {
	//            const S = rng.xof(s);
	//            let c = cshake128.create({ NISTfn: N, personalization: S });
	//            c.update(rng.xof(100));
	//            acc.update(c.xof(200));
	//            c = cshake128.create({ NISTfn: N, personalization: S });
	//            c.update(rng.xof(168));
	//            acc.update(c.xof(200));
	//            c = cshake128.create({ NISTfn: N, personalization: S });
	//            c.update(rng.xof(200));
	//            acc.update(c.xof(200));
	//        }
	//    }
	//    console.log(bytesToHex(acc.xof(32)));
	//
	cryptotest.TestAllImplementations(t, "sha3", func(t *testing.T) {
		t.Run("cSHAKE128", func(t *testing.T) {
			testCSHAKEAccumulated(t, NewCShake128, (1600-256)/8,
				"bb14f8657c6ec5403d0b0e2ef3d3393497e9d3b1a9a9e8e6c81dbaa5fd809252")
		})
		t.Run("cSHAKE256", func(t *testing.T) {
			testCSHAKEAccumulated(t, NewCShake256, (1600-512)/8,
				"0baaf9250c6e25f0c14ea5c7f9bfde54c8a922c8276437db28f3895bdf6eeeef")
		})
	})
}

func testCSHAKEAccumulated(t *testing.T, newCShake func(N, S []byte) *SHAKE, rate int64, exp string) {
	rnd := newCShake(nil, nil)
	acc := newCShake(nil, nil)
	for n := 0; n < 200; n++ {
		N := make([]byte, n)
		rnd.Read(N)
		for s := 0; s < 200; s++ {
			S := make([]byte, s)
			rnd.Read(S)

			c := newCShake(N, S)
			io.CopyN(c, rnd, 100 /* < rate */)
			io.CopyN(acc, c, 200)

			c.Reset()
			io.CopyN(c, rnd, rate)
			io.CopyN(acc, c, 200)

			c.Reset()
			io.CopyN(c, rnd, 200 /* > rate */)
			io.CopyN(acc, c, 200)
		}
	}
	if got := hex.EncodeToString(acc.Sum(nil)[:32]); got != exp {
		t.Errorf("got %s, want %s", got, exp)
	}
}

func TestCSHAKELargeS(t *testing.T) {
	cryptotest.TestAllImplementations(t, "sha3", testCSHAKELargeS)
}

func testCSHAKELargeS(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	// See https://go.dev/issue/66232.
	const s = (1<<32)/8 + 1000 // s * 8 > 2^32
	S := make([]byte, s)
	rnd := NewShake128()
	rnd.Read(S)
	c := NewCShake128(nil, S)
	io.CopyN(c, rnd, 1000)

	// Generated with pycryptodome@3.20.0
	//
	//    from Crypto.Hash import cSHAKE128
	//    rng = cSHAKE128.new()
	//    S = rng.read(536871912)
	//    c = cSHAKE128.new(custom=S)
	//    c.update(rng.read(1000))
	//    print(c.read(32).hex())
	//
	exp := "2cb9f237767e98f2614b8779cf096a52da9b3a849280bbddec820771ae529cf0"
	if got := hex.EncodeToString(c.Sum(nil)); got != exp {
		t.Errorf("got %s, want %s", got, exp)
	}
}

func TestMarshalUnmarshal(t *testing.T) {
	cryptotest.TestAllImplementations(t, "sha3", func(t *testing.T) {
		t.Run("SHA3-224", func(t *testing.T) { testMarshalUnmarshal(t, New224()) })
		t.Run("SHA3-256", func(t *testing.T) { testMarshalUnmarshal(t, New256()) })
		t.Run("SHA3-384", func(t *testing.T) { testMarshalUnmarshal(t, New384()) })
		t.Run("SHA3-512", func(t *testing.T) { testMarshalUnmarshal(t, New512()) })
		t.Run("SHAKE128", func(t *testing.T) { testMarshalUnmarshal(t, NewShake128()) })
		t.Run("SHAKE256", func(t *testing.T) { testMarshalUnmarshal(t, NewShake256()) })
		t.Run("cSHAKE128", func(t *testing.T) { testMarshalUnmarshal(t, NewCShake128([]byte("N"), []byte("S"))) })
		t.Run("cSHAKE256", func(t *testing.T) { testMarshalUnmarshal(t, NewCShake256([]byte("N"), []byte("S"))) })
		t.Run("Keccak-256", func(t *testing.T) { testMarshalUnmarshal(t, NewLegacyKeccak256()) })
		t.Run("Keccak-512", func(t *testing.T) { testMarshalUnmarshal(t, NewLegacyKeccak512()) })
	})
}

// TODO(filippo): move this to crypto/internal/cryptotest.
func testMarshalUnmarshal(t *testing.T, h fips.Hash) {
	buf := make([]byte, 200)
	rand.Read(buf)
	n := rand.Intn(200)
	h.Write(buf)
	want := h.Sum(nil)
	h.Reset()
	h.Write(buf[:n])
	b, err := h.(encoding.BinaryMarshaler).MarshalBinary()
	if err != nil {
		t.Errorf("MarshalBinary: %v", err)
	}
	h.Write(bytes.Repeat([]byte{0}, 200))
	if err := h.(encoding.BinaryUnmarshaler).UnmarshalBinary(b); err != nil {
		t.Errorf("UnmarshalBinary: %v", err)
	}
	h.Write(buf[n:])
	got := h.Sum(nil)
	if !bytes.Equal(got, want) {
		t.Errorf("got %x, want %x", got, want)
	}
}

// benchmarkHash tests the speed to hash num buffers of buflen each.
func benchmarkHash(b *testing.B, h fips.Hash, size, num int) {
	b.StopTimer()
	h.Reset()
	data := sequentialBytes(size)
	b.SetBytes(int64(size * num))
	b.StartTimer()

	var state []byte
	for i := 0; i < b.N; i++ {
		for j := 0; j < num; j++ {
			h.Write(data)
		}
		state = h.Sum(state[:0])
	}
	b.StopTimer()
	h.Reset()
}

// benchmarkShake is specialized to the Shake instances, which don't
// require a copy on reading output.
func benchmarkShake(b *testing.B, h *SHAKE, size, num int) {
	b.StopTimer()
	h.Reset()
	data := sequentialBytes(size)
	d := make([]byte, 32)

	b.SetBytes(int64(size * num))
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		h.Reset()
		for j := 0; j < num; j++ {
			h.Write(data)
		}
		h.Read(d)
	}
}

func BenchmarkSha3_512_MTU(b *testing.B) { benchmarkHash(b, New512(), 1350, 1) }
func BenchmarkSha3_384_MTU(b *testing.B) { benchmarkHash(b, New384(), 1350, 1) }
func BenchmarkSha3_256_MTU(b *testing.B) { benchmarkHash(b, New256(), 1350, 1) }
func BenchmarkSha3_224_MTU(b *testing.B) { benchmarkHash(b, New224(), 1350, 1) }

func BenchmarkShake128_MTU(b *testing.B)  { benchmarkShake(b, NewShake128(), 1350, 1) }
func BenchmarkShake256_MTU(b *testing.B)  { benchmarkShake(b, NewShake256(), 1350, 1) }
func BenchmarkShake256_16x(b *testing.B)  { benchmarkShake(b, NewShake256(), 16, 1024) }
func BenchmarkShake256_1MiB(b *testing.B) { benchmarkShake(b, NewShake256(), 1024, 1024) }

func BenchmarkSha3_512_1MiB(b *testing.B) { benchmarkHash(b, New512(), 1024, 1024) }

func Example_sum() {
	buf := []byte("some data to hash")
	// A hash needs to be 64 bytes long to have 256-bit collision resistance.
	h := make([]byte, 64)
	// Compute a 64-byte hash of buf and put it in h.
	ShakeSum256(h, buf)
	fmt.Printf("%x\n", h)
	// Output: 0f65fe41fc353e52c55667bb9e2b27bfcc8476f2c413e9437d272ee3194a4e3146d05ec04a25d16b8f577c19b82d16b1424c3e022e783d2b4da98de3658d363d
}

func Example_mac() {
	k := []byte("this is a secret key; you should generate a strong random key that's at least 32 bytes long")
	buf := []byte("and this is some data to authenticate")
	// A MAC with 32 bytes of output has 256-bit security strength -- if you use at least a 32-byte-long key.
	h := make([]byte, 32)
	d := NewShake256()
	// Write the key into the hash.
	d.Write(k)
	// Now write the data.
	d.Write(buf)
	// Read 32 bytes of output from the hash into h.
	d.Read(h)
	fmt.Printf("%x\n", h)
	// Output: 78de2974bd2711d5549ffd32b753ef0f5fa80a0db2556db60f0987eb8a9218ff
}

func ExampleNewCShake256() {
	out := make([]byte, 32)
	msg := []byte("The quick brown fox jumps over the lazy dog")

	// Example 1: Simple cshake
	c1 := NewCShake256([]byte("NAME"), []byte("Partition1"))
	c1.Write(msg)
	c1.Read(out)
	fmt.Println(hex.EncodeToString(out))

	// Example 2: Different customization string produces different digest
	c1 = NewCShake256([]byte("NAME"), []byte("Partition2"))
	c1.Write(msg)
	c1.Read(out)
	fmt.Println(hex.EncodeToString(out))

	// Example 3: Longer output length produces longer digest
	out = make([]byte, 64)
	c1 = NewCShake256([]byte("NAME"), []byte("Partition1"))
	c1.Write(msg)
	c1.Read(out)
	fmt.Println(hex.EncodeToString(out))

	// Example 4: Next read produces different result
	c1.Read(out)
	fmt.Println(hex.EncodeToString(out))

	// Output:
	//a90a4c6ca9af2156eba43dc8398279e6b60dcd56fb21837afe6c308fd4ceb05b
	//a8db03e71f3e4da5c4eee9d28333cdd355f51cef3c567e59be5beb4ecdbb28f0
	//a90a4c6ca9af2156eba43dc8398279e6b60dcd56fb21837afe6c308fd4ceb05b9dd98c6ee866ca7dc5a39d53e960f400bcd5a19c8a2d6ec6459f63696543a0d8
	//85e73a72228d08b46515553ca3a29d47df3047e5d84b12d6c2c63e579f4fd1105716b7838e92e981863907f434bfd4443c9e56ea09da998d2f9b47db71988109
}

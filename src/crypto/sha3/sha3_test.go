// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sha3_test

import (
	"bytes"
	"crypto/internal/cryptotest"
	. "crypto/sha3"
	"encoding/hex"
	"hash"
	"io"
	"math/rand"
	"strings"
	"testing"
)

const testString = "brekeccakkeccak koax koax"

// testDigests contains functions returning hash.Hash instances
// with output-length equal to the KAT length for SHA-3, Keccak
// and SHAKE instances.
var testDigests = map[string]func() *SHA3{
	"SHA3-224": New224,
	"SHA3-256": New256,
	"SHA3-384": New384,
	"SHA3-512": New512,
}

// testShakes contains functions that return *sha3.SHAKE instances for
// with output-length equal to the KAT length.
var testShakes = map[string]struct {
	constructor  func(N []byte, S []byte) *SHAKE
	defAlgoName  string
	defCustomStr string
}{
	// NewCSHAKE without customization produces same result as SHAKE
	"SHAKE128":  {NewCSHAKE128, "", ""},
	"SHAKE256":  {NewCSHAKE256, "", ""},
	"cSHAKE128": {NewCSHAKE128, "CSHAKE128", "CustomString"},
	"cSHAKE256": {NewCSHAKE256, "CSHAKE256", "CustomString"},
}

// decodeHex converts a hex-encoded string into a raw byte string.
func decodeHex(s string) []byte {
	b, err := hex.DecodeString(s)
	if err != nil {
		panic(err)
	}
	return b
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
	t.Run("NewSHAKE", func(t *testing.T) {
		if allocs := testing.AllocsPerRun(10, func() {
			h := NewSHAKE128()
			b := []byte("ABC")
			h.Write(b)
			out := make([]byte, 32)
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
	t.Run("SumSHAKE", func(t *testing.T) {
		if allocs := testing.AllocsPerRun(10, func() {
			b := []byte("ABC")
			out := SumSHAKE128(b, 10)
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
			testCSHAKEAccumulated(t, NewCSHAKE128, (1600-256)/8,
				"bb14f8657c6ec5403d0b0e2ef3d3393497e9d3b1a9a9e8e6c81dbaa5fd809252")
		})
		t.Run("cSHAKE256", func(t *testing.T) {
			testCSHAKEAccumulated(t, NewCSHAKE256, (1600-512)/8,
				"0baaf9250c6e25f0c14ea5c7f9bfde54c8a922c8276437db28f3895bdf6eeeef")
		})
	})
}

func testCSHAKEAccumulated(t *testing.T, newCSHAKE func(N, S []byte) *SHAKE, rate int64, exp string) {
	rnd := newCSHAKE(nil, nil)
	acc := newCSHAKE(nil, nil)
	for n := 0; n < 200; n++ {
		N := make([]byte, n)
		rnd.Read(N)
		for s := 0; s < 200; s++ {
			S := make([]byte, s)
			rnd.Read(S)

			c := newCSHAKE(N, S)
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
	out := make([]byte, 32)
	acc.Read(out)
	if got := hex.EncodeToString(out); got != exp {
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
	rnd := NewSHAKE128()
	rnd.Read(S)
	c := NewCSHAKE128(nil, S)
	io.CopyN(c, rnd, 1000)
	out := make([]byte, 32)
	c.Read(out)

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
	if got := hex.EncodeToString(out); got != exp {
		t.Errorf("got %s, want %s", got, exp)
	}
}

func TestMarshalUnmarshal(t *testing.T) {
	cryptotest.TestAllImplementations(t, "sha3", func(t *testing.T) {
		t.Run("SHA3-224", func(t *testing.T) { testMarshalUnmarshal(t, New224()) })
		t.Run("SHA3-256", func(t *testing.T) { testMarshalUnmarshal(t, New256()) })
		t.Run("SHA3-384", func(t *testing.T) { testMarshalUnmarshal(t, New384()) })
		t.Run("SHA3-512", func(t *testing.T) { testMarshalUnmarshal(t, New512()) })
		t.Run("SHAKE128", func(t *testing.T) { testMarshalUnmarshalSHAKE(t, NewSHAKE128()) })
		t.Run("SHAKE256", func(t *testing.T) { testMarshalUnmarshalSHAKE(t, NewSHAKE256()) })
		t.Run("cSHAKE128", func(t *testing.T) { testMarshalUnmarshalSHAKE(t, NewCSHAKE128([]byte("N"), []byte("S"))) })
		t.Run("cSHAKE256", func(t *testing.T) { testMarshalUnmarshalSHAKE(t, NewCSHAKE256([]byte("N"), []byte("S"))) })
	})
}

// TODO(filippo): move this to crypto/internal/cryptotest.
func testMarshalUnmarshal(t *testing.T, h *SHA3) {
	buf := make([]byte, 200)
	rand.Read(buf)
	n := rand.Intn(200)
	h.Write(buf)
	want := h.Sum(nil)
	h.Reset()
	h.Write(buf[:n])
	b, err := h.MarshalBinary()
	if err != nil {
		t.Errorf("MarshalBinary: %v", err)
	}
	h.Write(bytes.Repeat([]byte{0}, 200))
	if err := h.UnmarshalBinary(b); err != nil {
		t.Errorf("UnmarshalBinary: %v", err)
	}
	h.Write(buf[n:])
	got := h.Sum(nil)
	if !bytes.Equal(got, want) {
		t.Errorf("got %x, want %x", got, want)
	}
}

// TODO(filippo): move this to crypto/internal/cryptotest.
func testMarshalUnmarshalSHAKE(t *testing.T, h *SHAKE) {
	buf := make([]byte, 200)
	rand.Read(buf)
	n := rand.Intn(200)
	h.Write(buf)
	want := make([]byte, 32)
	h.Read(want)
	h.Reset()
	h.Write(buf[:n])
	b, err := h.MarshalBinary()
	if err != nil {
		t.Errorf("MarshalBinary: %v", err)
	}
	h.Write(bytes.Repeat([]byte{0}, 200))
	if err := h.UnmarshalBinary(b); err != nil {
		t.Errorf("UnmarshalBinary: %v", err)
	}
	h.Write(buf[n:])
	got := make([]byte, 32)
	h.Read(got)
	if !bytes.Equal(got, want) {
		t.Errorf("got %x, want %x", got, want)
	}
}

// benchmarkHash tests the speed to hash num buffers of buflen each.
func benchmarkHash(b *testing.B, h hash.Hash, size, num int) {
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

func BenchmarkShake128_MTU(b *testing.B)  { benchmarkShake(b, NewSHAKE128(), 1350, 1) }
func BenchmarkShake256_MTU(b *testing.B)  { benchmarkShake(b, NewSHAKE256(), 1350, 1) }
func BenchmarkShake256_16x(b *testing.B)  { benchmarkShake(b, NewSHAKE256(), 16, 1024) }
func BenchmarkShake256_1MiB(b *testing.B) { benchmarkShake(b, NewSHAKE256(), 1024, 1024) }

func BenchmarkSha3_512_1MiB(b *testing.B) { benchmarkHash(b, New512(), 1024, 1024) }

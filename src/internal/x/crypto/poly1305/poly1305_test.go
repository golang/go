// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poly1305

import (
	"encoding/hex"
	"flag"
	"testing"
	"unsafe"
)

var stressFlag = flag.Bool("stress", false, "run slow stress tests")

type test struct {
	in  string
	key string
	tag string
}

func (t *test) Input() []byte {
	in, err := hex.DecodeString(t.in)
	if err != nil {
		panic(err)
	}
	return in
}

func (t *test) Key() [32]byte {
	buf, err := hex.DecodeString(t.key)
	if err != nil {
		panic(err)
	}
	var key [32]byte
	copy(key[:], buf[:32])
	return key
}

func (t *test) Tag() [16]byte {
	buf, err := hex.DecodeString(t.tag)
	if err != nil {
		panic(err)
	}
	var tag [16]byte
	copy(tag[:], buf[:16])
	return tag
}

func testSum(t *testing.T, unaligned bool, sumImpl func(tag *[TagSize]byte, msg []byte, key *[32]byte)) {
	var tag [16]byte
	for i, v := range testData {
		in := v.Input()
		if unaligned {
			in = unalignBytes(in)
		}
		key := v.Key()
		sumImpl(&tag, in, &key)
		if tag != v.Tag() {
			t.Errorf("%d: expected %x, got %x", i, v.Tag(), tag[:])
		}
	}
}

func TestBurnin(t *testing.T) {
	// This test can be used to sanity-check significant changes. It can
	// take about many minutes to run, even on fast machines. It's disabled
	// by default.
	if !*stressFlag {
		t.Skip("skipping without -stress")
	}

	var key [32]byte
	var input [25]byte
	var output [16]byte

	for i := range key {
		key[i] = 1
	}
	for i := range input {
		input[i] = 2
	}

	for i := uint64(0); i < 1e10; i++ {
		Sum(&output, input[:], &key)
		copy(key[0:], output[:])
		copy(key[16:], output[:])
		copy(input[:], output[:])
		copy(input[16:], output[:])
	}

	const expected = "5e3b866aea0b636d240c83c428f84bfa"
	if got := hex.EncodeToString(output[:]); got != expected {
		t.Errorf("expected %s, got %s", expected, got)
	}
}

func TestSum(t *testing.T)                 { testSum(t, false, Sum) }
func TestSumUnaligned(t *testing.T)        { testSum(t, true, Sum) }
func TestSumGeneric(t *testing.T)          { testSum(t, false, sumGeneric) }
func TestSumGenericUnaligned(t *testing.T) { testSum(t, true, sumGeneric) }

func benchmark(b *testing.B, size int, unaligned bool) {
	var out [16]byte
	var key [32]byte
	in := make([]byte, size)
	if unaligned {
		in = unalignBytes(in)
	}
	b.SetBytes(int64(len(in)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Sum(&out, in, &key)
	}
}

func Benchmark64(b *testing.B)          { benchmark(b, 64, false) }
func Benchmark1K(b *testing.B)          { benchmark(b, 1024, false) }
func Benchmark64Unaligned(b *testing.B) { benchmark(b, 64, true) }
func Benchmark1KUnaligned(b *testing.B) { benchmark(b, 1024, true) }
func Benchmark2M(b *testing.B)          { benchmark(b, 2097152, true) }

func unalignBytes(in []byte) []byte {
	out := make([]byte, len(in)+1)
	if uintptr(unsafe.Pointer(&out[0]))&(unsafe.Alignof(uint32(0))-1) == 0 {
		out = out[1:]
	} else {
		out = out[:len(in)]
	}
	copy(out, in)
	return out
}

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzz

import (
	"bytes"
	"fmt"
	"os"
	"strconv"
	"testing"
)

func BenchmarkMutatorBytes(b *testing.B) {
	origEnv := os.Getenv("GODEBUG")
	defer func() { os.Setenv("GODEBUG", origEnv) }()
	os.Setenv("GODEBUG", fmt.Sprintf("%s,fuzzseed=123", origEnv))
	m := newMutator()

	for _, size := range []int{
		1,
		10,
		100,
		1000,
		10000,
		100000,
	} {
		b.Run(strconv.Itoa(size), func(b *testing.B) {
			buf := make([]byte, size)
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				// resize buffer to the correct shape and reset the PCG
				buf = buf[0:size]
				m.r = newPcgRand()
				m.mutate([]any{buf}, workerSharedMemSize)
			}
		})
	}
}

func BenchmarkMutatorString(b *testing.B) {
	origEnv := os.Getenv("GODEBUG")
	defer func() { os.Setenv("GODEBUG", origEnv) }()
	os.Setenv("GODEBUG", fmt.Sprintf("%s,fuzzseed=123", origEnv))
	m := newMutator()

	for _, size := range []int{
		1,
		10,
		100,
		1000,
		10000,
		100000,
	} {
		b.Run(strconv.Itoa(size), func(b *testing.B) {
			buf := make([]byte, size)
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				// resize buffer to the correct shape and reset the PCG
				buf = buf[0:size]
				m.r = newPcgRand()
				m.mutate([]any{string(buf)}, workerSharedMemSize)
			}
		})
	}
}

func BenchmarkMutatorAllBasicTypes(b *testing.B) {
	origEnv := os.Getenv("GODEBUG")
	defer func() { os.Setenv("GODEBUG", origEnv) }()
	os.Setenv("GODEBUG", fmt.Sprintf("%s,fuzzseed=123", origEnv))
	m := newMutator()

	types := []any{
		[]byte(""),
		string(""),
		false,
		float32(0),
		float64(0),
		int(0),
		int8(0),
		int16(0),
		int32(0),
		int64(0),
		uint8(0),
		uint16(0),
		uint32(0),
		uint64(0),
	}

	for _, t := range types {
		b.Run(fmt.Sprintf("%T", t), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				m.r = newPcgRand()
				m.mutate([]any{t}, workerSharedMemSize)
			}
		})
	}
}

func TestStringImmutability(t *testing.T) {
	v := []any{"hello"}
	m := newMutator()
	m.mutate(v, 1024)
	original := v[0].(string)
	originalCopy := make([]byte, len(original))
	copy(originalCopy, []byte(original))
	for i := 0; i < 25; i++ {
		m.mutate(v, 1024)
	}
	if !bytes.Equal([]byte(original), originalCopy) {
		t.Fatalf("string was mutated: got %x, want %x", []byte(original), originalCopy)
	}
}

func TestMutatorByteSliceValuesAreIndependent(t *testing.T) {
	m := &mutator{r: &mockRand{values: []int{
		0, 4, 0, 0, // Select argument 0, byteSliceBitFlip, byte 0, bit 0.
		1, 4, 0, 0, // Select argument 1, byteSliceBitFlip, byte 0, bit 0.
	}}}
	vals := []any{[]byte{1, 2, 3}, []byte{4, 5, 6}}
	for i := range vals {
		other := 1 - i
		before := bytes.Clone(vals[other].([]byte))
		m.mutate(vals, 1024)
		if after := vals[other].([]byte); !bytes.Equal(after, before) {
			t.Fatalf("mutating argument %d changed argument %d: %x -> %x", i, other, before, after)
		}
	}
	a, b := vals[0].([]byte), vals[1].([]byte)
	if &a[0] == &b[0] {
		t.Fatal("arguments share one backing array")
	}
}

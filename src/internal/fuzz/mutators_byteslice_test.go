// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzz

import (
	"bytes"
	"fmt"
	"testing"
)

type mockRand struct {
	values  []int
	counter int
	b       bool
}

func (mr *mockRand) uint32() uint32 {
	c := mr.values[mr.counter]
	mr.counter++
	return uint32(c)
}

func (mr *mockRand) intn(n int) int {
	c := mr.values[mr.counter]
	mr.counter++
	return c % n
}

func (mr *mockRand) uint32n(n uint32) uint32 {
	c := mr.values[mr.counter]
	mr.counter++
	return uint32(c) % n
}

func (mr *mockRand) bool() bool {
	b := mr.b
	mr.b = !mr.b
	return b
}

func (mr *mockRand) save(*uint64, *uint64) {
	panic("unimplemented")
}

func (mr *mockRand) restore(uint64, uint64) {
	panic("unimplemented")
}

func TestByteSliceMutators(t *testing.T) {
	for _, tc := range []struct {
		name     string
		mutator  func(*mutator, []byte) []byte
		randVals []int
		input    []byte
		expected []byte
	}{
		{
			name:     "byteSliceRemoveBytes",
			mutator:  byteSliceRemoveBytes,
			input:    []byte{1, 2, 3, 4},
			expected: []byte{4},
		},
		{
			name:     "byteSliceInsertRandomBytes",
			mutator:  byteSliceInsertRandomBytes,
			input:    make([]byte, 4, 8),
			expected: []byte{3, 4, 5, 0, 0, 0, 0},
		},
		{
			name:     "byteSliceDuplicateBytes",
			mutator:  byteSliceDuplicateBytes,
			input:    append(make([]byte, 0, 13), []byte{1, 2, 3, 4}...),
			expected: []byte{1, 1, 2, 3, 4, 2, 3, 4},
		},
		{
			name:     "byteSliceOverwriteBytes",
			mutator:  byteSliceOverwriteBytes,
			input:    []byte{1, 2, 3, 4},
			expected: []byte{1, 1, 3, 4},
		},
		{
			name:     "byteSliceBitFlip",
			mutator:  byteSliceBitFlip,
			input:    []byte{1, 2, 3, 4},
			expected: []byte{3, 2, 3, 4},
		},
		{
			name:     "byteSliceXORByte",
			mutator:  byteSliceXORByte,
			input:    []byte{1, 2, 3, 4},
			expected: []byte{3, 2, 3, 4},
		},
		{
			name:     "byteSliceSwapByte",
			mutator:  byteSliceSwapByte,
			input:    []byte{1, 2, 3, 4},
			expected: []byte{2, 1, 3, 4},
		},
		{
			name:     "byteSliceArithmeticUint8",
			mutator:  byteSliceArithmeticUint8,
			input:    []byte{1, 2, 3, 4},
			expected: []byte{255, 2, 3, 4},
		},
		{
			name:     "byteSliceArithmeticUint16",
			mutator:  byteSliceArithmeticUint16,
			input:    []byte{1, 2, 3, 4},
			expected: []byte{1, 3, 3, 4},
		},
		{
			name:     "byteSliceArithmeticUint32",
			mutator:  byteSliceArithmeticUint32,
			input:    []byte{1, 2, 3, 4},
			expected: []byte{2, 2, 3, 4},
		},
		{
			name:     "byteSliceArithmeticUint64",
			mutator:  byteSliceArithmeticUint64,
			input:    []byte{1, 2, 3, 4, 5, 6, 7, 8},
			expected: []byte{2, 2, 3, 4, 5, 6, 7, 8},
		},
		{
			name:     "byteSliceOverwriteInterestingUint8",
			mutator:  byteSliceOverwriteInterestingUint8,
			input:    []byte{1, 2, 3, 4},
			expected: []byte{255, 2, 3, 4},
		},
		{
			name:     "byteSliceOverwriteInterestingUint16",
			mutator:  byteSliceOverwriteInterestingUint16,
			input:    []byte{1, 2, 3, 4},
			expected: []byte{255, 127, 3, 4},
		},
		{
			name:     "byteSliceOverwriteInterestingUint32",
			mutator:  byteSliceOverwriteInterestingUint32,
			input:    []byte{1, 2, 3, 4},
			expected: []byte{250, 0, 0, 250},
		},
		{
			name:     "byteSliceInsertConstantBytes",
			mutator:  byteSliceInsertConstantBytes,
			input:    append(make([]byte, 0, 8), []byte{1, 2, 3, 4}...),
			expected: []byte{3, 3, 3, 1, 2, 3, 4},
		},
		{
			name:     "byteSliceOverwriteConstantBytes",
			mutator:  byteSliceOverwriteConstantBytes,
			input:    []byte{1, 2, 3, 4},
			expected: []byte{3, 3, 3, 4},
		},
		{
			name:     "byteSliceShuffleBytes",
			mutator:  byteSliceShuffleBytes,
			input:    []byte{1, 2, 3, 4},
			expected: []byte{2, 3, 1, 4},
		},
		{
			name:     "byteSliceSwapBytes",
			mutator:  byteSliceSwapBytes,
			randVals: []int{0, 2, 0, 2},
			input:    append(make([]byte, 0, 9), []byte{1, 2, 3, 4}...),
			expected: []byte{3, 2, 1, 4},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := &mockRand{values: []int{0, 1, 2, 3, 4, 5}}
			if tc.randVals != nil {
				r.values = tc.randVals
			}
			m := &mutator{r: r}
			b := tc.mutator(m, tc.input)
			if !bytes.Equal(b, tc.expected) {
				t.Errorf("got %x, want %x", b, tc.expected)
			}
		})
	}
}

func BenchmarkByteSliceMutators(b *testing.B) {
	tests := [...]struct {
		name    string
		mutator func(*mutator, []byte) []byte
	}{
		{"RemoveBytes", byteSliceRemoveBytes},
		{"InsertRandomBytes", byteSliceInsertRandomBytes},
		{"DuplicateBytes", byteSliceDuplicateBytes},
		{"OverwriteBytes", byteSliceOverwriteBytes},
		{"BitFlip", byteSliceBitFlip},
		{"XORByte", byteSliceXORByte},
		{"SwapByte", byteSliceSwapByte},
		{"ArithmeticUint8", byteSliceArithmeticUint8},
		{"ArithmeticUint16", byteSliceArithmeticUint16},
		{"ArithmeticUint32", byteSliceArithmeticUint32},
		{"ArithmeticUint64", byteSliceArithmeticUint64},
		{"OverwriteInterestingUint8", byteSliceOverwriteInterestingUint8},
		{"OverwriteInterestingUint16", byteSliceOverwriteInterestingUint16},
		{"OverwriteInterestingUint32", byteSliceOverwriteInterestingUint32},
		{"InsertConstantBytes", byteSliceInsertConstantBytes},
		{"OverwriteConstantBytes", byteSliceOverwriteConstantBytes},
		{"ShuffleBytes", byteSliceShuffleBytes},
		{"SwapBytes", byteSliceSwapBytes},
	}

	for _, tc := range tests {
		b.Run(tc.name, func(b *testing.B) {
			for size := 64; size <= 1024; size *= 2 {
				b.Run(fmt.Sprintf("%d", size), func(b *testing.B) {
					m := &mutator{r: newPcgRand()}
					input := make([]byte, size)
					for i := 0; i < b.N; i++ {
						tc.mutator(m, input)
					}
				})
			}
		})
	}
}

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"bytes"
	"math"
	"testing"
	"unsafe"
)

func binwireLE(x uint64, width int) []byte {
	out := make([]byte, width)
	for i := range out {
		out[i] = byte(x >> (8 * i))
	}
	return out
}

func binwireView[T any](s []T) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(s))), len(s)*int(unsafe.Sizeof(s[0])))
}

func TestBinaryHostLittleEndianMatchesMemory(t *testing.T) {
	probe := uint16(1)
	littleEndian := *(*byte)(unsafe.Pointer(&probe)) == 1
	if littleEndian != BinaryHostLittleEndian {
		t.Fatalf("BinaryHostLittleEndian = %v, but a uint16 holding 1 has its low byte first = %v", BinaryHostLittleEndian, littleEndian)
	}
}

func TestBinaryHostLittleEndianMemoryEqualsWire(t *testing.T) {
	if !BinaryHostLittleEndian {
		t.Skip("big-endian host keeps the per-element wire path")
	}
	ints := []int64{0, 1, -1, math.MaxInt64, math.MinInt64, 0x0102030405060708}
	var want []byte
	for _, v := range ints {
		want = append(want, binwireLE(uint64(v), 8)...)
	}
	if got := binwireView(ints); !bytes.Equal(got, want) {
		t.Errorf("int64 memory %x, wire %x", got, want)
	}
	uints := []uint64{0, 1, math.MaxUint64, 0x8000000000000000, 0x0807060504030201}
	want = want[:0]
	for _, v := range uints {
		want = append(want, binwireLE(v, 8)...)
	}
	if got := binwireView(uints); !bytes.Equal(got, want) {
		t.Errorf("uint64 memory %x, wire %x", got, want)
	}
	doubles := []float64{0, math.Copysign(0, -1), 1.5, math.Inf(1), math.Inf(-1), math.Float64frombits(0x7ff8000000000123)}
	want = want[:0]
	for _, v := range doubles {
		want = append(want, binwireLE(math.Float64bits(v), 8)...)
	}
	if got := binwireView(doubles); !bytes.Equal(got, want) {
		t.Errorf("float64 memory %x, wire %x", got, want)
	}
	singles := []float32{0, math.Float32frombits(0x80000000), 1.5, float32(math.Inf(1)), float32(math.Inf(-1)), math.Float32frombits(0x7fc00123)}
	want = want[:0]
	for _, v := range singles {
		want = append(want, binwireLE(uint64(math.Float32bits(v)), 4)...)
	}
	if got := binwireView(singles); !bytes.Equal(got, want) {
		t.Errorf("float32 memory %x, wire %x", got, want)
	}
}

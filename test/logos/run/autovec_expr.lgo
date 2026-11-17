// run -goexperiment simd

package main

import (
	"fmt"
	"slices"
)

type integer interface {
	~int8 | ~uint8 | ~int16 | ~uint16 | ~int32 | ~uint32 | ~int64 | ~uint64
}

type wide interface {
	~int16 | ~uint16 | ~int32 | ~uint32 | ~int64 | ~uint64
}

type unsignedWide interface {
	~uint16 | ~uint32 | ~uint64
}

type signedShift interface {
	~int16 | ~int32
}

type word uint32

//go:noinline
func negate[T integer](x []T) {
	for i := range x {
		x[i] = -x[i]
	}
}

//go:noinline
func complement[T integer](x []T) {
	for i := range x {
		x[i] = ^x[i]
	}
}

//go:noinline
func doubleBytes(x []uint8) {
	for i := range x {
		x[i] += x[i]
	}
}

//go:noinline
func left[T wide](x []T) {
	for i := range x {
		x[i] <<= 3
	}
}

//go:noinline
func unsignedRight[T unsignedWide](x []T) {
	for i := range x {
		x[i] >>= 3
	}
}

//go:noinline
func signedRight[T signedShift](x []T) {
	for i := range x {
		x[i] >>= 3
	}
}

//go:noinline
func compound[T wide](x []T) {
	for i := range x {
		x[i] = ^(-x[i]) + (x[i] << 3)
	}
}

// The reference has no loop and is deliberately a call in each iteration.
//
//go:noinline
func scalar[T integer](v T, op byte) T {
	switch op {
	case '-':
		return -v
	case '^':
		return ^v
	case 'd':
		return v + v
	case '<':
		return v << 3
	case '>':
		return v >> 3
	case 'c':
		return ^(-v) + (v << 3)
	}
	panic("invalid scalar expression")
}

func checkKernel[T integer](name string, bits uint, op byte, kernel func([]T)) {
	lanes := 128 / int(bits)
	mask := (uint64(1) << bits) - 1
	samples := []uint64{0, 1, mask, uint64(1) << (bits - 1), (uint64(1) << (bits - 1)) - 1, 0xfedcba9876543281}
	kernel(nil)
	for _, n := range []int{0, 1, lanes - 1, lanes, lanes + 1, 2*lanes - 1, 2 * lanes, 2*lanes + 1, 7*lanes + 3} {
		backing := make([]T, n+2)
		backing[0], backing[n+1] = 11, 23
		x := backing[1 : n+1]
		want := make([]T, n)
		for i := range x {
			x[i] = T(samples[i%len(samples)])
			want[i] = scalar(x[i], op)
		}
		kernel(x)
		if !slices.Equal(x, want) || backing[0] != 11 || backing[n+1] != 23 {
			panic(fmt.Sprintf("%s length %d: got %v want %v; canaries %v/%v", name, n, x, want, backing[0], backing[n+1]))
		}
	}
}

// Runtime counts remain scalar. Include oversized shifts to catch accidental
// count masking when the expression matcher changes.
//
//go:noinline
func variableRight[T integer](x []T, count uint64) {
	for i := range x {
		x[i] >>= count
	}
}

//go:noinline
func scalarRight[T integer](v T, count uint64) T { return v >> count }

func checkShiftCounts[T integer](bits uint) {
	for _, count := range []uint64{0, uint64(bits - 1), uint64(bits), uint64(bits + 1), ^uint64(0)} {
		x := make([]T, 35)
		want := make([]T, len(x))
		for i := range x {
			v := uint64(i+1)*0x9e3779b97f4a7c15 ^ 0x8000000000000081
			x[i] = T(v)
			want[i] = scalarRight(x[i], count)
		}
		variableRight(x, count)
		if !slices.Equal(x, want) {
			panic(fmt.Sprintf("width %d shift %d: got %v want %v", bits, count, x, want))
		}
	}
}

func main() {
	checkKernel("int8 negate", 8, '-', negate[int8])
	checkKernel("int8 complement", 8, '^', complement[int8])
	checkShiftCounts[int8](8)
	checkKernel("uint8 negate", 8, '-', negate[uint8])
	checkKernel("uint8 complement", 8, '^', complement[uint8])
	checkShiftCounts[uint8](8)
	checkKernel("int16 negate", 16, '-', negate[int16])
	checkKernel("int16 complement", 16, '^', complement[int16])
	checkKernel("int16 left", 16, '<', left[int16])
	checkKernel("int16 compound", 16, 'c', compound[int16])
	checkKernel("int16 right", 16, '>', signedRight[int16])
	checkShiftCounts[int16](16)
	checkKernel("uint16 negate", 16, '-', negate[uint16])
	checkKernel("uint16 complement", 16, '^', complement[uint16])
	checkKernel("uint16 left", 16, '<', left[uint16])
	checkKernel("uint16 compound", 16, 'c', compound[uint16])
	checkKernel("uint16 right", 16, '>', unsignedRight[uint16])
	checkShiftCounts[uint16](16)
	checkKernel("int32 negate", 32, '-', negate[int32])
	checkKernel("int32 complement", 32, '^', complement[int32])
	checkKernel("int32 left", 32, '<', left[int32])
	checkKernel("int32 compound", 32, 'c', compound[int32])
	checkKernel("int32 right", 32, '>', signedRight[int32])
	checkShiftCounts[int32](32)
	checkKernel("uint32 negate", 32, '-', negate[uint32])
	checkKernel("uint32 complement", 32, '^', complement[uint32])
	checkKernel("uint32 left", 32, '<', left[uint32])
	checkKernel("uint32 compound", 32, 'c', compound[uint32])
	checkKernel("uint32 right", 32, '>', unsignedRight[uint32])
	checkShiftCounts[uint32](32)
	checkKernel("int64 negate", 64, '-', negate[int64])
	checkKernel("int64 complement", 64, '^', complement[int64])
	checkKernel("int64 left", 64, '<', left[int64])
	checkKernel("int64 compound", 64, 'c', compound[int64])
	checkShiftCounts[int64](64)
	checkKernel("uint64 negate", 64, '-', negate[uint64])
	checkKernel("uint64 complement", 64, '^', complement[uint64])
	checkKernel("uint64 left", 64, '<', left[uint64])
	checkKernel("uint64 compound", 64, 'c', compound[uint64])
	checkKernel("uint64 right", 64, '>', unsignedRight[uint64])
	checkShiftCounts[uint64](64)
	checkKernel("uint8 double", 8, 'd', doubleBytes)
	checkKernel("word complement", 32, '^', complement[word])
	checkKernel("word compound", 32, 'c', compound[word])
}

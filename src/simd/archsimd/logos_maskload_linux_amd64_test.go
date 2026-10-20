// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && linux && amd64

package archsimd_test

import (
	"simd/archsimd"
	"syscall"
	"testing"
	"unsafe"
)

// pageEndSlice returns a slice of n elements of T whose last element ends
// exactly at the end of a mapped page; the next page is PROT_NONE, so a
// load that touches any byte past the slice faults. The release function
// unmaps both pages.
func pageEndSlice[T any](t *testing.T, n int) (s []T, release func()) {
	t.Helper()
	page := syscall.Getpagesize()
	mem, err := syscall.Mmap(-1, 0, 2*page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatalf("mmap: %v", err)
	}
	if err := syscall.Mprotect(mem[page:], syscall.PROT_NONE); err != nil {
		syscall.Munmap(mem)
		t.Fatalf("mprotect: %v", err)
	}
	var zero T
	size := int(unsafe.Sizeof(zero))
	s = unsafe.Slice((*T)(unsafe.Pointer(&mem[page-n*size])), n)
	return s, func() {
		if err := syscall.Munmap(mem); err != nil {
			t.Errorf("munmap: %v", err)
		}
	}
}

// checkSliceMasked is the #22 attack at the archsimd layer: for every
// length from 0 through one full vector, load reads a slice that ends at
// the last byte of a page whose successor is PROT_NONE, with a mask that
// enables exactly the elements of the slice. An over-read faults the
// process; a correct load returns the elements followed by zeroes.
func checkSliceMasked[T comparable](t *testing.T, lanes int, load func(s []T) []T, fill func(i int) T) {
	t.Helper()
	var zero T
	for n := 0; n <= lanes; n++ {
		s, release := pageEndSlice[T](t, n)
		for i := range s {
			s[i] = fill(i)
		}
		got := load(s)
		if len(got) != lanes {
			t.Fatalf("n=%d: vector holds %d lanes, want %d", n, len(got), lanes)
		}
		for i, v := range got {
			want := zero
			if i < n {
				want = s[i]
			}
			if v != want {
				t.Errorf("n=%d: lane %d is %v, want %v", n, i, v, want)
			}
		}
		release()
	}
}

func intFill[T ~int8 | ~uint8 | ~int16 | ~uint16 | ~int32 | ~uint32 | ~int64 | ~uint64](i int) T {
	return T(i + 1)
}

func floatFill[T ~float32 | ~float64](i int) T {
	return T(i) + 1.5
}

// mask32x8 enables the first k of 8 32-bit lanes.
func mask32x8(k int) archsimd.Mask32x8 {
	table := [16]int32{-1, -1, -1, -1, -1, -1, -1, -1}
	var zero archsimd.Int32x8
	return zero.Greater(archsimd.LoadInt32x8(table[8-k:]))
}

// mask64x4 enables the first k of 4 64-bit lanes.
func mask64x4(k int) archsimd.Mask64x4 {
	table := [8]int64{-1, -1, -1, -1}
	var zero archsimd.Int64x4
	return zero.Greater(archsimd.LoadInt64x4(table[4-k:]))
}

func TestLoadSliceMaskedPageEnd(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("AVX2 is not available")
	}
	t.Run("Float32x8", func(t *testing.T) {
		checkSliceMasked(t, 8, func(s []float32) []float32 {
			var out [8]float32
			archsimd.LoadFloat32x8SliceMasked(s, mask32x8(len(s))).StoreArray(&out)
			return out[:]
		}, floatFill[float32])
	})
	t.Run("Int32x8", func(t *testing.T) {
		checkSliceMasked(t, 8, func(s []int32) []int32 {
			var out [8]int32
			archsimd.LoadInt32x8SliceMasked(s, mask32x8(len(s))).StoreArray(&out)
			return out[:]
		}, intFill[int32])
	})
	t.Run("Uint32x8", func(t *testing.T) {
		checkSliceMasked(t, 8, func(s []uint32) []uint32 {
			var out [8]uint32
			archsimd.LoadUint32x8SliceMasked(s, mask32x8(len(s))).StoreArray(&out)
			return out[:]
		}, intFill[uint32])
	})
	t.Run("Float64x4", func(t *testing.T) {
		checkSliceMasked(t, 4, func(s []float64) []float64 {
			var out [4]float64
			archsimd.LoadFloat64x4SliceMasked(s, mask64x4(len(s))).StoreArray(&out)
			return out[:]
		}, floatFill[float64])
	})
	t.Run("Int64x4", func(t *testing.T) {
		checkSliceMasked(t, 4, func(s []int64) []int64 {
			var out [4]int64
			archsimd.LoadInt64x4SliceMasked(s, mask64x4(len(s))).StoreArray(&out)
			return out[:]
		}, intFill[int64])
	})
	t.Run("Uint64x4", func(t *testing.T) {
		checkSliceMasked(t, 4, func(s []uint64) []uint64 {
			var out [4]uint64
			archsimd.LoadUint64x4SliceMasked(s, mask64x4(len(s))).StoreArray(&out)
			return out[:]
		}, intFill[uint64])
	})
}

// checkSliceMaskedAllOnes is the SPEC Sec 15.2 attack at the archsimd
// layer: load reads a slice that ends at the last byte of a page whose
// successor is PROT_NONE, with an all-ones mask. Lanes at or past len(s)
// must read no memory, so the load must not fault, and disabled lanes
// must produce zero. A loader that trusts the caller mask over-reads
// into the guard page and faults the process.
func checkSliceMaskedAllOnes[T comparable](t *testing.T, lanes int, load func(s []T) []T, fill func(i int) T) {
	t.Helper()
	var zero T
	for n := 0; n <= lanes; n++ {
		s, release := pageEndSlice[T](t, n)
		for i := range s {
			s[i] = fill(i)
		}
		got := load(s)
		if len(got) != lanes {
			t.Fatalf("n=%d: vector holds %d lanes, want %d", n, len(got), lanes)
		}
		for i, v := range got {
			want := zero
			if i < n {
				want = s[i]
			}
			if v != want {
				t.Errorf("n=%d: lane %d is %v, want %v", n, i, v, want)
			}
		}
		release()
	}
}

func TestLoadSliceMaskedAllOnesPageEnd(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("AVX2 is not available")
	}
	t.Run("Float32x8", func(t *testing.T) {
		checkSliceMaskedAllOnes(t, 8, func(s []float32) []float32 {
			var out [8]float32
			archsimd.LoadFloat32x8SliceMasked(s, mask32x8(8)).StoreArray(&out)
			return out[:]
		}, floatFill[float32])
	})
	t.Run("Int32x8", func(t *testing.T) {
		checkSliceMaskedAllOnes(t, 8, func(s []int32) []int32 {
			var out [8]int32
			archsimd.LoadInt32x8SliceMasked(s, mask32x8(8)).StoreArray(&out)
			return out[:]
		}, intFill[int32])
	})
	t.Run("Uint32x8", func(t *testing.T) {
		checkSliceMaskedAllOnes(t, 8, func(s []uint32) []uint32 {
			var out [8]uint32
			archsimd.LoadUint32x8SliceMasked(s, mask32x8(8)).StoreArray(&out)
			return out[:]
		}, intFill[uint32])
	})
	t.Run("Float64x4", func(t *testing.T) {
		checkSliceMaskedAllOnes(t, 4, func(s []float64) []float64 {
			var out [4]float64
			archsimd.LoadFloat64x4SliceMasked(s, mask64x4(4)).StoreArray(&out)
			return out[:]
		}, floatFill[float64])
	})
	t.Run("Int64x4", func(t *testing.T) {
		checkSliceMaskedAllOnes(t, 4, func(s []int64) []int64 {
			var out [4]int64
			archsimd.LoadInt64x4SliceMasked(s, mask64x4(4)).StoreArray(&out)
			return out[:]
		}, intFill[int64])
	})
	t.Run("Uint64x4", func(t *testing.T) {
		checkSliceMaskedAllOnes(t, 4, func(s []uint64) []uint64 {
			var out [4]uint64
			archsimd.LoadUint64x4SliceMasked(s, mask64x4(4)).StoreArray(&out)
			return out[:]
		}, intFill[uint64])
	})
}

func TestLoadSliceMaskedAllOnesPageEnd512(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("AVX512 is not available")
	}
	t.Run("Int16x32", func(t *testing.T) {
		checkSliceMaskedAllOnes(t, 32, func(s []int16) []int16 {
			var out [32]int16
			archsimd.LoadInt16x32SliceMasked(s, archsimd.Mask16x32FromBits(^uint32(0))).StoreArray(&out)
			return out[:]
		}, intFill[int16])
	})
	t.Run("Uint16x32", func(t *testing.T) {
		checkSliceMaskedAllOnes(t, 32, func(s []uint16) []uint16 {
			var out [32]uint16
			archsimd.LoadUint16x32SliceMasked(s, archsimd.Mask16x32FromBits(^uint32(0))).StoreArray(&out)
			return out[:]
		}, intFill[uint16])
	})
	t.Run("Int8x64", func(t *testing.T) {
		checkSliceMaskedAllOnes(t, 64, func(s []int8) []int8 {
			var out [64]int8
			archsimd.LoadInt8x64SliceMasked(s, archsimd.Mask8x64FromBits(^uint64(0))).StoreArray(&out)
			return out[:]
		}, intFill[int8])
	})
	t.Run("Uint8x64", func(t *testing.T) {
		checkSliceMaskedAllOnes(t, 64, func(s []uint8) []uint8 {
			var out [64]uint8
			archsimd.LoadUint8x64SliceMasked(s, archsimd.Mask8x64FromBits(^uint64(0))).StoreArray(&out)
			return out[:]
		}, intFill[uint8])
	})
}

func TestLoadSliceMaskedPageEnd512(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("AVX512 is not available")
	}
	t.Run("Int16x32", func(t *testing.T) {
		checkSliceMasked(t, 32, func(s []int16) []int16 {
			var out [32]int16
			archsimd.LoadInt16x32SliceMasked(s, archsimd.Mask16x32FromBits(^uint32(0)>>(32-len(s)))).StoreArray(&out)
			return out[:]
		}, intFill[int16])
	})
	t.Run("Uint16x32", func(t *testing.T) {
		checkSliceMasked(t, 32, func(s []uint16) []uint16 {
			var out [32]uint16
			archsimd.LoadUint16x32SliceMasked(s, archsimd.Mask16x32FromBits(^uint32(0)>>(32-len(s)))).StoreArray(&out)
			return out[:]
		}, intFill[uint16])
	})
	t.Run("Int8x64", func(t *testing.T) {
		checkSliceMasked(t, 64, func(s []int8) []int8 {
			var out [64]int8
			archsimd.LoadInt8x64SliceMasked(s, archsimd.Mask8x64FromBits(^uint64(0)>>(64-len(s)))).StoreArray(&out)
			return out[:]
		}, intFill[int8])
	})
	t.Run("Uint8x64", func(t *testing.T) {
		checkSliceMasked(t, 64, func(s []uint8) []uint8 {
			var out [64]uint8
			archsimd.LoadUint8x64SliceMasked(s, archsimd.Mask8x64FromBits(^uint64(0)>>(64-len(s)))).StoreArray(&out)
			return out[:]
		}, intFill[uint8])
	})
}

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package scan

import (
	"internal/abi"
	"internal/runtime/gc"
	"math/bits"
	"simd"
	"unsafe"
)

func FilterNilAVX512(bufp *uintptr, n int32) (cnt int32) {
	scanned := 0
	buf := unsafe.Slice((*uint64)(unsafe.Pointer(bufp)), int(n))
	// Use the widest vector
	var zeros simd.Uint64x8
	for ; scanned+8 <= int(n); scanned += 8 {
		v := simd.LoadUint64x8Slice(buf[scanned:])
		m := v.NotEqual(zeros)
		v.Compress(m).StoreSlice(buf[cnt:])
		// Count the mask bits
		mbits := uint64(m.ToBits())
		mbits &= 0xFF // Only the lower 8 bits are meaningful.
		nonNilCnt := bits.OnesCount64(mbits)
		cnt += int32(nonNilCnt)
	}
	// Scalar code to clean up tails.
	for i := scanned; i < int(n); i++ {
		if buf[i] != 0 {
			buf[cnt] = buf[i]
			cnt++
		}
	}
	return
}

func ScanSpanPackedAVX512(mem unsafe.Pointer, bufp *uintptr, objMarks *gc.ObjMask, sizeClass uintptr, ptrMask *gc.PtrMask) (count int32) {
	return FilterNilAVX512(bufp, scanSpanPackedAVX512(mem, bufp, objMarks, sizeClass, ptrMask))
}

func scanSpanPackedAVX512(mem unsafe.Pointer, buf *uintptr, objDarts *gc.ObjMask, sizeClass uintptr, ptrMask *gc.PtrMask) (count int32) {
	// Expand the grey object mask into a grey word mask
	m1, m2 := gcExpandersAVX512[sizeClass](abi.NoEscape(unsafe.Pointer(objDarts)))
	// Load the pointer mask
	ptrm := unsafe.Pointer(ptrMask)
	m3 := simd.LoadUint64x8((*[8]uint64)(ptrm))
	m4 := simd.LoadUint64x8((*[8]uint64)(unsafe.Pointer(uintptr(ptrm) + 64)))

	masks := [128]uint8{}
	counts := [128]uint8{}
	// Combine the grey word mask with the pointer mask to get the scan mask
	m1m3 := m1.And(m3).AsUint8x64()
	m2m4 := m2.And(m4).AsUint8x64()
	m1m3.Store((*[64]uint8)(unsafe.Pointer(&masks[0])))
	m2m4.Store((*[64]uint8)(unsafe.Pointer(&masks[64])))
	// Now each bit of m1m3 and m2m4 represents one word of the span.
	// Thus, each byte covers 64 bytes of memory, which is also how
	// much we can fix in a ZMM register.
	//
	// We do a load/compress for each 64 byte frame.
	//
	// counts = Number of memory words to scan in each 64 byte frame
	// TODO: Right now the type casting is done via memory, is it possible to
	// workaround these stores and loads and keep them in register?
	m1m3.OnesCount().Store((*[64]uint8)(unsafe.Pointer(&counts[0])))
	m2m4.OnesCount().Store((*[64]uint8)(unsafe.Pointer(&counts[64])))

	// Loop over the 64 byte frames in this span.
	// TODO: is there a way to PCALIGN this loop?
	for i := range 128 {
		mv := masks[i]
		// Skip empty frames.
		if mv == 0 {
			continue
		}
		// Load the 64 byte frame.
		m := simd.Mask64x8FromBits(mv)
		ptrs := simd.LoadUint64x8((*[8]uint64)(unsafe.Pointer(uintptr(mem) + uintptr(i*64))))
		// Collect just the pointers from the greyed objects into the scan buffer,
		// i.e., copy the word indices in the mask from Z1 into contiguous memory.
		ptrs.Compress(m).Store((*[8]uint64)(unsafe.Pointer(uintptr(unsafe.Pointer(buf)) + uintptr(count*8))))
		// Advance the scan buffer position by the number of pointers.
		count += int32(counts[i])
	}
	simd.ClearAVXUpperBits()
	return
}

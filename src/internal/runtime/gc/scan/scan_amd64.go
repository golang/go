// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scan

import (
	"internal/cpu"
	"internal/goexperiment"
	"internal/runtime/gc"
	"unsafe"
)

func ScanSpanPacked(mem unsafe.Pointer, bufp *uintptr, objMarks *gc.ObjMask, sizeClass uintptr, ptrMask *gc.PtrMask) (count int32) {
	if CanAVX512() {
		if goexperiment.SIMD {
			return ScanSpanPackedAVX512(mem, bufp, objMarks, sizeClass, ptrMask)
		} else {
			return ScanSpanPackedAVX512Asm(mem, bufp, objMarks, sizeClass, ptrMask)
		}
	}
	panic("not implemented")
}

func ScanSpanPackedAsm(mem unsafe.Pointer, bufp *uintptr, objMarks *gc.ObjMask, sizeClass uintptr, ptrMask *gc.PtrMask) (count int32) {
	if CanAVX512() {
		return ScanSpanPackedAVX512Asm(mem, bufp, objMarks, sizeClass, ptrMask)
	}
	panic("not implemented")
}

func HasFastScanSpanPacked() bool {
	return avx512ScanPackedReqsMet
}

// -- AVX512 --

func CanAVX512() bool {
	return avx512ScanPackedReqsMet
}

func ScanSpanPackedAVX512Asm(mem unsafe.Pointer, bufp *uintptr, objMarks *gc.ObjMask, sizeClass uintptr, ptrMask *gc.PtrMask) (count int32) {
	return FilterNil(bufp, scanSpanPackedAVX512Asm(mem, bufp, objMarks, sizeClass, ptrMask))
}

//go:noescape
func scanSpanPackedAVX512Asm(mem unsafe.Pointer, bufp *uintptr, objMarks *gc.ObjMask, sizeClass uintptr, ptrMask *gc.PtrMask) (count int32)

var avx512ScanPackedReqsMet = cpu.X86.HasAVX512VL &&
	cpu.X86.HasAVX512BW &&
	cpu.X86.HasGFNI &&
	cpu.X86.HasAVX512BITALG &&
	cpu.X86.HasAVX512VBMI

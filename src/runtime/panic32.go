// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build 386 || arm || mips || mipsle

package runtime

import (
	"internal/abi"
	"internal/runtime/sys"
)

// Additional index/slice error paths for 32-bit platforms.
// Used when the high word of a 64-bit index is not zero.

// failures in the comparisons for s[x], 0 <= x < y (y == len(s))
func goPanicExtendIndex(hi int, lo uint, y int) {
	panicCheck1(sys.GetCallerPC(), "index out of range")
	panic(boundsError{x: int64(hi)<<32 + int64(lo), signed: true, y: y, code: abi.BoundsIndex})
}
func goPanicExtendIndexU(hi uint, lo uint, y int) {
	panicCheck1(sys.GetCallerPC(), "index out of range")
	panic(boundsError{x: int64(hi)<<32 + int64(lo), signed: false, y: y, code: abi.BoundsIndex})
}

// failures in the comparisons for s[:x], 0 <= x <= y (y == len(s) or cap(s))
func goPanicExtendSliceAlen(hi int, lo uint, y int) {
	panicCheck1(sys.GetCallerPC(), "slice bounds out of range")
	panic(boundsError{x: int64(hi)<<32 + int64(lo), signed: true, y: y, code: abi.BoundsSliceAlen})
}
func goPanicExtendSliceAlenU(hi uint, lo uint, y int) {
	panicCheck1(sys.GetCallerPC(), "slice bounds out of range")
	panic(boundsError{x: int64(hi)<<32 + int64(lo), signed: false, y: y, code: abi.BoundsSliceAlen})
}
func goPanicExtendSliceAcap(hi int, lo uint, y int) {
	panicCheck1(sys.GetCallerPC(), "slice bounds out of range")
	panic(boundsError{x: int64(hi)<<32 + int64(lo), signed: true, y: y, code: abi.BoundsSliceAcap})
}
func goPanicExtendSliceAcapU(hi uint, lo uint, y int) {
	panicCheck1(sys.GetCallerPC(), "slice bounds out of range")
	panic(boundsError{x: int64(hi)<<32 + int64(lo), signed: false, y: y, code: abi.BoundsSliceAcap})
}

// failures in the comparisons for s[x:y], 0 <= x <= y
func goPanicExtendSliceB(hi int, lo uint, y int) {
	panicCheck1(sys.GetCallerPC(), "slice bounds out of range")
	panic(boundsError{x: int64(hi)<<32 + int64(lo), signed: true, y: y, code: abi.BoundsSliceB})
}
func goPanicExtendSliceBU(hi uint, lo uint, y int) {
	panicCheck1(sys.GetCallerPC(), "slice bounds out of range")
	panic(boundsError{x: int64(hi)<<32 + int64(lo), signed: false, y: y, code: abi.BoundsSliceB})
}

// failures in the comparisons for s[::x], 0 <= x <= y (y == len(s) or cap(s))
func goPanicExtendSlice3Alen(hi int, lo uint, y int) {
	panicCheck1(sys.GetCallerPC(), "slice bounds out of range")
	panic(boundsError{x: int64(hi)<<32 + int64(lo), signed: true, y: y, code: abi.BoundsSlice3Alen})
}
func goPanicExtendSlice3AlenU(hi uint, lo uint, y int) {
	panicCheck1(sys.GetCallerPC(), "slice bounds out of range")
	panic(boundsError{x: int64(hi)<<32 + int64(lo), signed: false, y: y, code: abi.BoundsSlice3Alen})
}
func goPanicExtendSlice3Acap(hi int, lo uint, y int) {
	panicCheck1(sys.GetCallerPC(), "slice bounds out of range")
	panic(boundsError{x: int64(hi)<<32 + int64(lo), signed: true, y: y, code: abi.BoundsSlice3Acap})
}
func goPanicExtendSlice3AcapU(hi uint, lo uint, y int) {
	panicCheck1(sys.GetCallerPC(), "slice bounds out of range")
	panic(boundsError{x: int64(hi)<<32 + int64(lo), signed: false, y: y, code: abi.BoundsSlice3Acap})
}

// failures in the comparisons for s[:x:y], 0 <= x <= y
func goPanicExtendSlice3B(hi int, lo uint, y int) {
	panicCheck1(sys.GetCallerPC(), "slice bounds out of range")
	panic(boundsError{x: int64(hi)<<32 + int64(lo), signed: true, y: y, code: abi.BoundsSlice3B})
}
func goPanicExtendSlice3BU(hi uint, lo uint, y int) {
	panicCheck1(sys.GetCallerPC(), "slice bounds out of range")
	panic(boundsError{x: int64(hi)<<32 + int64(lo), signed: false, y: y, code: abi.BoundsSlice3B})
}

// failures in the comparisons for s[x:y:], 0 <= x <= y
func goPanicExtendSlice3C(hi int, lo uint, y int) {
	panicCheck1(sys.GetCallerPC(), "slice bounds out of range")
	panic(boundsError{x: int64(hi)<<32 + int64(lo), signed: true, y: y, code: abi.BoundsSlice3C})
}
func goPanicExtendSlice3CU(hi uint, lo uint, y int) {
	panicCheck1(sys.GetCallerPC(), "slice bounds out of range")
	panic(boundsError{x: int64(hi)<<32 + int64(lo), signed: false, y: y, code: abi.BoundsSlice3C})
}

// Implemented in assembly, as they take arguments in registers.
// Declared here to mark them as ABIInternal.
func panicExtendIndex(hi int, lo uint, y int)
func panicExtendIndexU(hi uint, lo uint, y int)
func panicExtendSliceAlen(hi int, lo uint, y int)
func panicExtendSliceAlenU(hi uint, lo uint, y int)
func panicExtendSliceAcap(hi int, lo uint, y int)
func panicExtendSliceAcapU(hi uint, lo uint, y int)
func panicExtendSliceB(hi int, lo uint, y int)
func panicExtendSliceBU(hi uint, lo uint, y int)
func panicExtendSlice3Alen(hi int, lo uint, y int)
func panicExtendSlice3AlenU(hi uint, lo uint, y int)
func panicExtendSlice3Acap(hi int, lo uint, y int)
func panicExtendSlice3AcapU(hi uint, lo uint, y int)
func panicExtendSlice3B(hi int, lo uint, y int)
func panicExtendSlice3BU(hi uint, lo uint, y int)
func panicExtendSlice3C(hi int, lo uint, y int)
func panicExtendSlice3CU(hi uint, lo uint, y int)

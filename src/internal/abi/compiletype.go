// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package abi

// These functions are the build-time version of the Go type data structures.

// Their contents must be kept in sync with their definitions.
// Because the host and target type sizes can differ, the compiler and
// linker cannot use the host information that they might get from
// either unsafe.Sizeof and Alignof, nor runtime, reflect, or reflectlite.

// CommonSize returns sizeof(Type) for a compilation target with a given ptrSize
func CommonSize(ptrSize int) int { return 4*ptrSize + 8 + 8 }

// StructFieldSize returns sizeof(StructField) for a compilation target with a given ptrSize
func StructFieldSize(ptrSize int) int { return 3 * ptrSize }

// UncommonSize returns sizeof(UncommonType).  This currently does not depend on ptrSize.
// This exported function is in an internal package, so it may change to depend on ptrSize in the future.
func UncommonSize() uint64 { return 4 + 2 + 2 + 4 + 4 }

// IMethodSize returns sizeof(IMethod) for a compilation target with a given ptrSize
func IMethodSize(ptrSize int) int { return 4 + 4 }

// TFlagOff returns the offset of Type.TFlag for a compilation target with a given ptrSize
func TFlagOff(ptrSize int) int { return 2*ptrSize + 4 }

// Offset is for computing offsets of type data structures at compile/link time;
// the target platform may not be the host platform.  Its state includes the
// current offset, necessary alignment for the sequence of types, and the size
// of pointers and alignment of slices, interfaces, and strings (this is for tearing-
// resistant access to these types, if/when that is supported).
type Offset struct {
	off        uint64 // the current offset
	align      uint8  // the required alignmentof the container
	ptrSize    uint8  // the size of a pointer in bytes
	sliceAlign uint8  // the alignment of slices (and interfaces and strings)
}

// NewOffset returns a new Offset with offset 0 and alignment 1.
func NewOffset(ptrSize uint8, twoWordAlignSlices bool) Offset {
	if twoWordAlignSlices {
		return Offset{off: 0, align: 1, ptrSize: ptrSize, sliceAlign: 2 * ptrSize}
	}
	return Offset{off: 0, align: 1, ptrSize: ptrSize, sliceAlign: ptrSize}
}

func assertIsAPowerOfTwo(x uint8) {
	if x == 0 {
		panic("Zero is not a power of two")
	}
	if x&-x == x {
		return
	}
	panic("Not a power of two")
}

// InitializedOffset returns a new Offset with specified offset, alignment, pointer size, and slice alignment.
func InitializedOffset(off int, align uint8, ptrSize uint8, twoWordAlignSlices bool) Offset {
	assertIsAPowerOfTwo(align)
	o0 := NewOffset(ptrSize, twoWordAlignSlices)
	o0.off = uint64(off)
	o0.align = align
	return o0
}

func (o Offset) align_(a uint8) Offset {
	o.off = (o.off + uint64(a) - 1) & ^(uint64(a) - 1)
	if o.align < a {
		o.align = a
	}
	return o
}

// Align returns the offset obtained by aligning offset to a multiple of a.
// a must be a power of two.
func (o Offset) Align(a uint8) Offset {
	assertIsAPowerOfTwo(a)
	return o.align_(a)
}

// plus returns the offset obtained by appending a power-of-2-sized-and-aligned object to o.
func (o Offset) plus(x uint64) Offset {
	o = o.align_(uint8(x))
	o.off += x
	return o
}

// D8 returns the offset obtained by appending an 8-bit field to o.
func (o Offset) D8() Offset {
	return o.plus(1)
}

// D16 returns the offset obtained by appending a 16-bit field to o.
func (o Offset) D16() Offset {
	return o.plus(2)
}

// D32 returns the offset obtained by appending a 32-bit field to o.
func (o Offset) D32() Offset {
	return o.plus(4)
}

// D64 returns the offset obtained by appending a 64-bit field to o.
func (o Offset) D64() Offset {
	return o.plus(8)
}

// D64 returns the offset obtained by appending a pointer field to o.
func (o Offset) P() Offset {
	if o.ptrSize == 0 {
		panic("This offset has no defined pointer size")
	}
	return o.plus(uint64(o.ptrSize))
}

// Slice returns the offset obtained by appending a slice field to o.
func (o Offset) Slice() Offset {
	o = o.align_(o.sliceAlign)
	o.off += 3 * uint64(o.ptrSize)
	// There's been discussion of whether slices should be 2-word aligned to allow
	// use of aligned 2-word load/store to prevent tearing, this is future proofing.
	// In general, for purposes of struct layout (and very likely default C layout
	// compatibility) the "size" of a Go type is rounded up to its alignment.
	return o.Align(o.sliceAlign)
}

// String returns the offset obtained by appending a string field to o.
func (o Offset) String() Offset {
	o = o.align_(o.sliceAlign)
	o.off += 2 * uint64(o.ptrSize)
	return o // We "know" it needs no further alignment
}

// Interface returns the offset obtained by appending an interface field to o.
func (o Offset) Interface() Offset {
	o = o.align_(o.sliceAlign)
	o.off += 2 * uint64(o.ptrSize)
	return o // We "know" it needs no further alignment
}

// Offset returns the struct-aligned offset (size) of o.
// This is at least as large as the current internal offset; it may be larger.
func (o Offset) Offset() uint64 {
	return o.Align(o.align).off
}

func (o Offset) PlusUncommon() Offset {
	o.off += UncommonSize()
	return o
}

// CommonOffset returns the Offset to the data after the common portion of type data structures.
func CommonOffset(ptrSize int, twoWordAlignSlices bool) Offset {
	return InitializedOffset(CommonSize(ptrSize), uint8(ptrSize), uint8(ptrSize), twoWordAlignSlices)
}

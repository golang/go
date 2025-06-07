// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

import "unsafe"

// This file contains code generation tests related to the handling of
// slice types.

// ------------------ //
//      Clear         //
// ------------------ //

// Issue #5373 optimize memset idiom
// Some of the clears get inlined, see #56997

func SliceClear(s []int) []int {
	// amd64:`.*memclrNoHeapPointers`
	// ppc64x:`.*memclrNoHeapPointers`
	for i := range s {
		s[i] = 0
	}
	return s
}

func SliceClearPointers(s []*int) []*int {
	// amd64:`.*memclrHasPointers`
	// ppc64x:`.*memclrHasPointers`
	for i := range s {
		s[i] = nil
	}
	return s
}

// ------------------ //
//      Extension     //
// ------------------ //

// Issue #21266 - avoid makeslice in append(x, make([]T, y)...)

func SliceExtensionConst(s []int) []int {
	// amd64:-`.*runtime\.memclrNoHeapPointers`
	// amd64:-`.*runtime\.makeslice`
	// amd64:-`.*runtime\.panicmakeslicelen`
	// amd64:"MOVUPS\tX15"
	// loong64:-`.*runtime\.memclrNoHeapPointers`
	// ppc64x:-`.*runtime\.memclrNoHeapPointers`
	// ppc64x:-`.*runtime\.makeslice`
	// ppc64x:-`.*runtime\.panicmakeslicelen`
	return append(s, make([]int, 1<<2)...)
}

func SliceExtensionConstInt64(s []int) []int {
	// amd64:-`.*runtime\.memclrNoHeapPointers`
	// amd64:-`.*runtime\.makeslice`
	// amd64:-`.*runtime\.panicmakeslicelen`
	// amd64:"MOVUPS\tX15"
	// loong64:-`.*runtime\.memclrNoHeapPointers`
	// ppc64x:-`.*runtime\.memclrNoHeapPointers`
	// ppc64x:-`.*runtime\.makeslice`
	// ppc64x:-`.*runtime\.panicmakeslicelen`
	return append(s, make([]int, int64(1<<2))...)
}

func SliceExtensionConstUint64(s []int) []int {
	// amd64:-`.*runtime\.memclrNoHeapPointers`
	// amd64:-`.*runtime\.makeslice`
	// amd64:-`.*runtime\.panicmakeslicelen`
	// amd64:"MOVUPS\tX15"
	// loong64:-`.*runtime\.memclrNoHeapPointers`
	// ppc64x:-`.*runtime\.memclrNoHeapPointers`
	// ppc64x:-`.*runtime\.makeslice`
	// ppc64x:-`.*runtime\.panicmakeslicelen`
	return append(s, make([]int, uint64(1<<2))...)
}

func SliceExtensionConstUint(s []int) []int {
	// amd64:-`.*runtime\.memclrNoHeapPointers`
	// amd64:-`.*runtime\.makeslice`
	// amd64:-`.*runtime\.panicmakeslicelen`
	// amd64:"MOVUPS\tX15"
	// loong64:-`.*runtime\.memclrNoHeapPointers`
	// ppc64x:-`.*runtime\.memclrNoHeapPointers`
	// ppc64x:-`.*runtime\.makeslice`
	// ppc64x:-`.*runtime\.panicmakeslicelen`
	return append(s, make([]int, uint(1<<2))...)
}

// On ppc64x and loong64 continue to use memclrNoHeapPointers
// for sizes >= 512.
func SliceExtensionConst512(s []int) []int {
	// amd64:-`.*runtime\.memclrNoHeapPointers`
	// loong64:`.*runtime\.memclrNoHeapPointers`
	// ppc64x:`.*runtime\.memclrNoHeapPointers`
	return append(s, make([]int, 1<<9)...)
}

func SliceExtensionPointer(s []*int, l int) []*int {
	// amd64:`.*runtime\.memclrHasPointers`
	// amd64:-`.*runtime\.makeslice`
	// ppc64x:`.*runtime\.memclrHasPointers`
	// ppc64x:-`.*runtime\.makeslice`
	return append(s, make([]*int, l)...)
}

func SliceExtensionVar(s []byte, l int) []byte {
	// amd64:`.*runtime\.memclrNoHeapPointers`
	// amd64:-`.*runtime\.makeslice`
	// ppc64x:`.*runtime\.memclrNoHeapPointers`
	// ppc64x:-`.*runtime\.makeslice`
	return append(s, make([]byte, l)...)
}

func SliceExtensionVarInt64(s []byte, l int64) []byte {
	// amd64:`.*runtime\.memclrNoHeapPointers`
	// amd64:-`.*runtime\.makeslice`
	// amd64:`.*runtime\.panicmakeslicelen`
	return append(s, make([]byte, l)...)
}

func SliceExtensionVarUint64(s []byte, l uint64) []byte {
	// amd64:`.*runtime\.memclrNoHeapPointers`
	// amd64:-`.*runtime\.makeslice`
	// amd64:`.*runtime\.panicmakeslicelen`
	return append(s, make([]byte, l)...)
}

func SliceExtensionVarUint(s []byte, l uint) []byte {
	// amd64:`.*runtime\.memclrNoHeapPointers`
	// amd64:-`.*runtime\.makeslice`
	// amd64:`.*runtime\.panicmakeslicelen`
	return append(s, make([]byte, l)...)
}

func SliceExtensionInt64(s []int, l64 int64) []int {
	// 386:`.*runtime\.makeslice`
	// 386:-`.*runtime\.memclr`
	return append(s, make([]int, l64)...)
}

// ------------------ //
//      Make+Copy     //
// ------------------ //

// Issue #26252 - avoid memclr for make+copy

func SliceMakeCopyLen(s []int) []int {
	// amd64:`.*runtime\.mallocgc`
	// amd64:`.*runtime\.memmove`
	// amd64:-`.*runtime\.makeslice`
	// ppc64x:`.*runtime\.mallocgc`
	// ppc64x:`.*runtime\.memmove`
	// ppc64x:-`.*runtime\.makeslice`
	a := make([]int, len(s))
	copy(a, s)
	return a
}

func SliceMakeCopyLenPtr(s []*int) []*int {
	// amd64:`.*runtime\.makeslicecopy`
	// amd64:-`.*runtime\.makeslice\(`
	// amd64:-`.*runtime\.typedslicecopy
	// ppc64x:`.*runtime\.makeslicecopy`
	// ppc64x:-`.*runtime\.makeslice\(`
	// ppc64x:-`.*runtime\.typedslicecopy
	a := make([]*int, len(s))
	copy(a, s)
	return a
}

func SliceMakeCopyConst(s []int) []int {
	// amd64:`.*runtime\.makeslicecopy`
	// amd64:-`.*runtime\.makeslice\(`
	// amd64:-`.*runtime\.memmove`
	a := make([]int, 4)
	copy(a, s)
	return a
}

func SliceMakeCopyConstPtr(s []*int) []*int {
	// amd64:`.*runtime\.makeslicecopy`
	// amd64:-`.*runtime\.makeslice\(`
	// amd64:-`.*runtime\.typedslicecopy
	a := make([]*int, 4)
	copy(a, s)
	return a
}

func SliceMakeCopyNoOptNoDeref(s []*int) []*int {
	a := new([]*int)
	// amd64:-`.*runtime\.makeslicecopy`
	// amd64:`.*runtime\.makeslice\(`
	*a = make([]*int, 4)
	// amd64:-`.*runtime\.makeslicecopy`
	// amd64:`.*runtime\.typedslicecopy`
	copy(*a, s)
	return *a
}

func SliceMakeCopyNoOptNoVar(s []*int) []*int {
	a := make([][]*int, 1)
	// amd64:-`.*runtime\.makeslicecopy`
	// amd64:`.*runtime\.makeslice\(`
	a[0] = make([]*int, 4)
	// amd64:-`.*runtime\.makeslicecopy`
	// amd64:`.*runtime\.typedslicecopy`
	copy(a[0], s)
	return a[0]
}

func SliceMakeCopyNoOptBlank(s []*int) []*int {
	var a []*int
	// amd64:-`.*runtime\.makeslicecopy`
	_ = make([]*int, 4)
	// amd64:-`.*runtime\.makeslicecopy`
	// amd64:`.*runtime\.typedslicecopy`
	copy(a, s)
	return a
}

func SliceMakeCopyNoOptNoMake(s []*int) []*int {
	// amd64:-`.*runtime\.makeslicecopy`
	// amd64:-`.*runtime\.objectnew`
	a := *new([]*int)
	// amd64:-`.*runtime\.makeslicecopy`
	// amd64:`.*runtime\.typedslicecopy`
	copy(a, s)
	return a
}

func SliceMakeCopyNoOptNoHeapAlloc(s []*int) int {
	// amd64:-`.*runtime\.makeslicecopy`
	a := make([]*int, 4)
	// amd64:-`.*runtime\.makeslicecopy`
	// amd64:`.*runtime\.typedslicecopy`
	copy(a, s)
	return cap(a)
}

func SliceMakeCopyNoOptNoCap(s []*int) []*int {
	// amd64:-`.*runtime\.makeslicecopy`
	// amd64:`.*runtime\.makeslice\(`
	a := make([]*int, 0, 4)
	// amd64:-`.*runtime\.makeslicecopy`
	// amd64:`.*runtime\.typedslicecopy`
	copy(a, s)
	return a
}

func SliceMakeCopyNoOptNoCopy(s []*int) []*int {
	copy := func(x, y []*int) {}
	// amd64:-`.*runtime\.makeslicecopy`
	// amd64:`.*runtime\.makeslice\(`
	a := make([]*int, 4)
	// amd64:-`.*runtime\.makeslicecopy`
	copy(a, s)
	return a
}

func SliceMakeCopyNoOptWrongOrder(s []*int) []*int {
	// amd64:-`.*runtime\.makeslicecopy`
	// amd64:`.*runtime\.makeslice\(`
	a := make([]*int, 4)
	// amd64:`.*runtime\.typedslicecopy`
	// amd64:-`.*runtime\.makeslicecopy`
	copy(s, a)
	return a
}

func SliceMakeCopyNoOptWrongAssign(s []*int) []*int {
	var a []*int
	// amd64:-`.*runtime\.makeslicecopy`
	// amd64:`.*runtime\.makeslice\(`
	s = make([]*int, 4)
	// amd64:`.*runtime\.typedslicecopy`
	// amd64:-`.*runtime\.makeslicecopy`
	copy(a, s)
	return s
}

func SliceMakeCopyNoOptCopyLength(s []*int) (int, []*int) {
	// amd64:-`.*runtime\.makeslicecopy`
	// amd64:`.*runtime\.makeslice\(`
	a := make([]*int, 4)
	// amd64:`.*runtime\.typedslicecopy`
	// amd64:-`.*runtime\.makeslicecopy`
	n := copy(a, s)
	return n, a
}

func SliceMakeCopyNoOptSelfCopy(s []*int) []*int {
	// amd64:-`.*runtime\.makeslicecopy`
	// amd64:`.*runtime\.makeslice\(`
	a := make([]*int, 4)
	// amd64:`.*runtime\.typedslicecopy`
	// amd64:-`.*runtime\.makeslicecopy`
	copy(a, a)
	return a
}

func SliceMakeCopyNoOptTargetReference(s []*int) []*int {
	// amd64:-`.*runtime\.makeslicecopy`
	// amd64:`.*runtime\.makeslice\(`
	a := make([]*int, 4)
	// amd64:`.*runtime\.typedslicecopy`
	// amd64:-`.*runtime\.makeslicecopy`
	copy(a, s[:len(a)])
	return a
}

func SliceMakeCopyNoOptCap(s []int) []int {
	// amd64:-`.*runtime\.makeslicecopy`
	// amd64:`.*runtime\.makeslice\(`
	a := make([]int, len(s), 9)
	// amd64:-`.*runtime\.makeslicecopy`
	// amd64:`.*runtime\.memmove`
	copy(a, s)
	return a
}

func SliceMakeCopyNoMemmoveDifferentLen(s []int) []int {
	// amd64:`.*runtime\.makeslicecopy`
	// amd64:-`.*runtime\.memmove`
	a := make([]int, len(s)-1)
	// amd64:-`.*runtime\.memmove`
	copy(a, s)
	return a
}

func SliceMakeEmptyPointerToZerobase() []int {
	// amd64:`LEAQ.+runtime\.zerobase`
	// amd64:-`.*runtime\.makeslice`
	return make([]int, 0)
}

// ---------------------- //
//   Nil check of &s[0]   //
// ---------------------- //
// See issue 30366
func SliceNilCheck(s []int) {
	p := &s[0]
	// amd64:-`TESTB`
	_ = *p
}

// ---------------------- //
//   Init slice literal   //
// ---------------------- //
// See issue 21561
func InitSmallSliceLiteral() []int {
	// amd64:`MOVQ\t[$]42`
	return []int{42}
}

func InitNotSmallSliceLiteral() []int {
	// amd64:`LEAQ\t.*stmp_`
	return []int{
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
		42,
	}
}

// --------------------------------------- //
//   Test PPC64 SUBFCconst folding rules   //
//   triggered by slice operations.        //
// --------------------------------------- //

func SliceWithConstCompare(a []int, b int) []int {
	var c []int = []int{1, 2, 3, 4, 5}
	if b+len(a) < len(c) {
		// ppc64x:-"NEG"
		return c[b:]
	}
	return a
}

func SliceWithSubtractBound(a []int, b int) []int {
	// ppc64x:"SUBC",-"NEG"
	return a[(3 - b):]
}

// --------------------------------------- //
//   Code generation for unsafe.Slice      //
// --------------------------------------- //

func Slice1(p *byte, i int) []byte {
	// amd64:-"MULQ"
	return unsafe.Slice(p, i)
}
func Slice0(p *struct{}, i int) []struct{} {
	// amd64:-"MULQ"
	return unsafe.Slice(p, i)
}

// --------------------------------------- //
//   Code generation for slice bounds      //
//   checking comparison                   //
// --------------------------------------- //

func SlicePut(a []byte, c uint8) []byte {
	// arm64:`CBZ\tR1`
	a[0] = c
	// arm64:`CMP\t\$1, R1`
	a = a[1:]
	a[0] = c
	// arm64:`CMP\t\$2, R1`
	a = a[1:]
	a[0] = c
	a = a[1:]
	return a
}

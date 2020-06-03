// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// This file contains code generation tests related to the handling of
// slice types.

// ------------------ //
//      Clear         //
// ------------------ //

// Issue #5373 optimize memset idiom

func SliceClear(s []int) []int {
	// amd64:`.*memclrNoHeapPointers`
	for i := range s {
		s[i] = 0
	}
	return s
}

func SliceClearPointers(s []*int) []*int {
	// amd64:`.*memclrHasPointers`
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
	// amd64:`.*runtime\.memclrNoHeapPointers`
	// amd64:-`.*runtime\.makeslice`
	// amd64:-`.*runtime\.panicmakeslicelen`
	return append(s, make([]int, 1<<2)...)
}

func SliceExtensionConstInt64(s []int) []int {
	// amd64:`.*runtime\.memclrNoHeapPointers`
	// amd64:-`.*runtime\.makeslice`
	// amd64:-`.*runtime\.panicmakeslicelen`
	return append(s, make([]int, int64(1<<2))...)
}

func SliceExtensionConstUint64(s []int) []int {
	// amd64:`.*runtime\.memclrNoHeapPointers`
	// amd64:-`.*runtime\.makeslice`
	// amd64:-`.*runtime\.panicmakeslicelen`
	return append(s, make([]int, uint64(1<<2))...)
}

func SliceExtensionConstUint(s []int) []int {
	// amd64:`.*runtime\.memclrNoHeapPointers`
	// amd64:-`.*runtime\.makeslice`
	// amd64:-`.*runtime\.panicmakeslicelen`
	return append(s, make([]int, uint(1<<2))...)
}

func SliceExtensionPointer(s []*int, l int) []*int {
	// amd64:`.*runtime\.memclrHasPointers`
	// amd64:-`.*runtime\.makeslice`
	return append(s, make([]*int, l)...)
}

func SliceExtensionVar(s []byte, l int) []byte {
	// amd64:`.*runtime\.memclrNoHeapPointers`
	// amd64:-`.*runtime\.makeslice`
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
	a := make([]int, len(s))
	copy(a, s)
	return a
}

func SliceMakeCopyLenPtr(s []*int) []*int {
	// amd64:`.*runtime\.makeslicecopy`
	// amd64:-`.*runtime\.makeslice\(`
	// amd64:-`.*runtime\.typedslicecopy
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
	// amd64:`MOVQ\t.*autotmp_`
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

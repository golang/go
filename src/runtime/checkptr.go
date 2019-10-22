// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

type ptrAlignError struct {
	ptr  unsafe.Pointer
	elem *_type
	n    uintptr
}

func (e ptrAlignError) RuntimeError() {}

func (e ptrAlignError) Error() string {
	return "runtime error: unsafe pointer conversion"
}

func checkptrAlignment(p unsafe.Pointer, elem *_type, n uintptr) {
	// Check that (*[n]elem)(p) is appropriately aligned.
	// TODO(mdempsky): What about fieldAlign?
	if uintptr(p)&(uintptr(elem.align)-1) != 0 {
		panic(ptrAlignError{p, elem, n})
	}

	// Check that (*[n]elem)(p) doesn't straddle multiple heap objects.
	if size := n * elem.size; size > 1 && checkptrBase(p) != checkptrBase(add(p, size-1)) {
		panic(ptrAlignError{p, elem, n})
	}
}

type ptrArithError struct {
	ptr       unsafe.Pointer
	originals []unsafe.Pointer
}

func (e ptrArithError) RuntimeError() {}

func (e ptrArithError) Error() string {
	return "runtime error: unsafe pointer arithmetic"
}

func checkptrArithmetic(p unsafe.Pointer, originals []unsafe.Pointer) {
	if 0 < uintptr(p) && uintptr(p) < minLegalPointer {
		panic(ptrArithError{p, originals})
	}

	// Check that if the computed pointer p points into a heap
	// object, then one of the original pointers must have pointed
	// into the same object.
	base := checkptrBase(p)
	if base == 0 {
		return
	}

	for _, original := range originals {
		if base == checkptrBase(original) {
			return
		}
	}

	panic(ptrArithError{p, originals})
}

func checkptrBase(p unsafe.Pointer) uintptr {
	base, _, _ := findObject(uintptr(p), 0, 0)
	// TODO(mdempsky): If base == 0, then check if p points to the
	// stack or a global variable.
	return base
}

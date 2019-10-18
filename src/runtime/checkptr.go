// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

type ptrAlign struct {
	ptr  unsafe.Pointer
	elem *_type
}

func checkptrAlignment(p unsafe.Pointer, elem *_type) {
	// Check that (*T)(p) is appropriately aligned.
	// TODO(mdempsky): What about fieldAlign?
	if uintptr(p)&(uintptr(elem.align)-1) != 0 {
		panic(ptrAlign{p, elem})
	}

	// Check that (*T)(p) doesn't straddle multiple heap objects.
	if elem.size != 1 && checkptrBase(p) != checkptrBase(add(p, elem.size-1)) {
		panic(ptrAlign{p, elem})
	}
}

type ptrArith struct {
	ptr       unsafe.Pointer
	originals []unsafe.Pointer
}

func checkptrArithmetic(p unsafe.Pointer, originals []unsafe.Pointer) {
	if 0 < uintptr(p) && uintptr(p) < minLegalPointer {
		panic(ptrArith{p, originals})
	}

	base := checkptrBase(p)
	if base == 0 {
		return
	}

	for _, original := range originals {
		if base == checkptrBase(original) {
			return
		}
	}

	panic(ptrArith{p, originals})
}

func checkptrBase(p unsafe.Pointer) uintptr {
	base, _, _ := findObject(uintptr(p), 0, 0)
	// TODO(mdempsky): If base == 0, then check if p points to the
	// stack or a global variable.
	return base
}

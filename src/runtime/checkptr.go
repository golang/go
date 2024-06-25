// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

func checkptrAlignment(p unsafe.Pointer, elem *_type, n uintptr) {
	// nil pointer is always suitably aligned (#47430).
	if p == nil {
		return
	}

	// Check that (*[n]elem)(p) is appropriately aligned.
	// Note that we allow unaligned pointers if the types they point to contain
	// no pointers themselves. See issue 37298.
	// TODO(mdempsky): What about fieldAlign?
	if elem.Pointers() && uintptr(p)&(uintptr(elem.Align_)-1) != 0 {
		throw("checkptr: misaligned pointer conversion")
	}

	// Check that (*[n]elem)(p) doesn't straddle multiple heap objects.
	// TODO(mdempsky): Fix #46938 so we don't need to worry about overflow here.
	if checkptrStraddles(p, n*elem.Size_) {
		throw("checkptr: converted pointer straddles multiple allocations")
	}
}

// checkptrStraddles reports whether the first size-bytes of memory
// addressed by ptr is known to straddle more than one Go allocation.
func checkptrStraddles(ptr unsafe.Pointer, size uintptr) bool {
	if size <= 1 {
		return false
	}

	// Check that add(ptr, size-1) won't overflow. This avoids the risk
	// of producing an illegal pointer value (assuming ptr is legal).
	if uintptr(ptr) >= -(size - 1) {
		return true
	}
	end := add(ptr, size-1)

	// TODO(mdempsky): Detect when [ptr, end] contains Go allocations,
	// but neither ptr nor end point into one themselves.

	return checkptrBase(ptr) != checkptrBase(end)
}

func checkptrArithmetic(p unsafe.Pointer, originals []unsafe.Pointer) {
	if uintptr(p) < minLegalPointer {
		return
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

	throw("checkptr: pointer arithmetic result points to invalid allocation")
}

// checkptrBase returns the base address for the allocation containing
// the address p.
//
// Importantly, if p1 and p2 point into the same variable, then
// checkptrBase(p1) == checkptrBase(p2). However, the converse/inverse
// is not necessarily true as allocations can have trailing padding,
// and multiple variables may be packed into a single allocation.
//
// checkptrBase should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/bytedance/sonic
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname checkptrBase
func checkptrBase(p unsafe.Pointer) uintptr {
	// stack
	if gp := getg(); gp.stack.lo <= uintptr(p) && uintptr(p) < gp.stack.hi {
		// TODO(mdempsky): Walk the stack to identify the
		// specific stack frame or even stack object that p
		// points into.
		//
		// In the mean time, use "1" as a pseudo-address to
		// represent the stack. This is an invalid address on
		// all platforms, so it's guaranteed to be distinct
		// from any of the addresses we might return below.
		return 1
	}

	// heap (must check after stack because of #35068)
	if base, _, _ := findObject(uintptr(p), 0, 0); base != 0 {
		return base
	}

	// data or bss
	for _, datap := range activeModules() {
		if datap.data <= uintptr(p) && uintptr(p) < datap.edata {
			return datap.data
		}
		if datap.bss <= uintptr(p) && uintptr(p) < datap.ebss {
			return datap.bss
		}
	}

	return 0
}

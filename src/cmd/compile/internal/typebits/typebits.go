// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typebits

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/bitvec"
	"cmd/compile/internal/types"
)

// NOTE: The bitmap for a specific type t could be cached in t after
// the first run and then simply copied into bv at the correct offset
// on future calls with the same type t.
func Set(t *types.Type, off int64, bv bitvec.BitVec) {
	if uint8(t.Alignment()) > 0 && off&int64(uint8(t.Alignment())-1) != 0 {
		base.Fatalf("typebits.Set: invalid initial alignment: type %v has alignment %d, but offset is %v", t, uint8(t.Alignment()), off)
	}
	if !t.HasPointers() {
		// Note: this case ensures that pointers to go:notinheap types
		// are not considered pointers by garbage collection and stack copying.
		return
	}

	switch t.Kind() {
	case types.TPTR, types.TUNSAFEPTR, types.TFUNC, types.TCHAN, types.TMAP:
		if off&int64(types.PtrSize-1) != 0 {
			base.Fatalf("typebits.Set: invalid alignment, %v", t)
		}
		bv.Set(int32(off / int64(types.PtrSize))) // pointer

	case types.TSTRING:
		// struct { byte *str; intgo len; }
		if off&int64(types.PtrSize-1) != 0 {
			base.Fatalf("typebits.Set: invalid alignment, %v", t)
		}
		bv.Set(int32(off / int64(types.PtrSize))) //pointer in first slot

	case types.TINTER:
		// struct { Itab *tab;	void *data; }
		// or, when isnilinter(t)==true:
		// struct { Type *type; void *data; }
		if off&int64(types.PtrSize-1) != 0 {
			base.Fatalf("typebits.Set: invalid alignment, %v", t)
		}
		// The first word of an interface is a pointer, but we don't
		// treat it as such.
		// 1. If it is a non-empty interface, the pointer points to an itab
		//    which is always in persistentalloc space.
		// 2. If it is an empty interface, the pointer points to a _type.
		//   a. If it is a compile-time-allocated type, it points into
		//      the read-only data section.
		//   b. If it is a reflect-allocated type, it points into the Go heap.
		//      Reflect is responsible for keeping a reference to
		//      the underlying type so it won't be GCd.
		// If we ever have a moving GC, we need to change this for 2b (as
		// well as scan itabs to update their itab._type fields).
		bv.Set(int32(off/int64(types.PtrSize) + 1)) // pointer in second slot

	case types.TSLICE:
		// struct { byte *array; uintgo len; uintgo cap; }
		if off&int64(types.PtrSize-1) != 0 {
			base.Fatalf("typebits.Set: invalid TARRAY alignment, %v", t)
		}
		bv.Set(int32(off / int64(types.PtrSize))) // pointer in first slot (BitsPointer)

	case types.TARRAY:
		elt := t.Elem()
		if elt.Size() == 0 {
			// Short-circuit for #20739.
			break
		}
		for i := int64(0); i < t.NumElem(); i++ {
			Set(elt, off, bv)
			off += elt.Size()
		}

	case types.TSTRUCT:
		for _, f := range t.Fields().Slice() {
			Set(f.Type, off+f.Offset, bv)
		}

	default:
		base.Fatalf("typebits.Set: unexpected type, %v", t)
	}
}

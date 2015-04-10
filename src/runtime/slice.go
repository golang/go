// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"unsafe"
)

type slice struct {
	array unsafe.Pointer
	len   int
	cap   int
}

// TODO: take uintptrs instead of int64s?
func makeslice(t *slicetype, len64, cap64 int64) slice {
	// NOTE: The len > MaxMem/elemsize check here is not strictly necessary,
	// but it produces a 'len out of range' error instead of a 'cap out of range' error
	// when someone does make([]T, bignumber). 'cap out of range' is true too,
	// but since the cap is only being supplied implicitly, saying len is clearer.
	// See issue 4085.
	len := int(len64)
	if len64 < 0 || int64(len) != len64 || t.elem.size > 0 && uintptr(len) > _MaxMem/uintptr(t.elem.size) {
		panic(errorString("makeslice: len out of range"))
	}
	cap := int(cap64)
	if cap < len || int64(cap) != cap64 || t.elem.size > 0 && uintptr(cap) > _MaxMem/uintptr(t.elem.size) {
		panic(errorString("makeslice: cap out of range"))
	}
	p := newarray(t.elem, uintptr(cap))
	return slice{p, len, cap}
}

func growslice(t *slicetype, old slice, n int) slice {
	if n < 1 {
		panic(errorString("growslice: invalid n"))
	}

	cap := old.cap + n
	if cap < old.cap || t.elem.size > 0 && uintptr(cap) > _MaxMem/uintptr(t.elem.size) {
		panic(errorString("growslice: cap out of range"))
	}

	if raceenabled {
		callerpc := getcallerpc(unsafe.Pointer(&t))
		racereadrangepc(old.array, uintptr(old.len*int(t.elem.size)), callerpc, funcPC(growslice))
	}

	et := t.elem
	if et.size == 0 {
		// append should not create a slice with nil pointer but non-zero len.
		// We assume that append doesn't need to preserve old.array in this case.
		return slice{unsafe.Pointer(&zerobase), old.len, cap}
	}

	newcap := old.cap
	if newcap+newcap < cap {
		newcap = cap
	} else {
		for {
			if old.len < 1024 {
				newcap += newcap
			} else {
				newcap += newcap / 4
			}
			if newcap >= cap {
				break
			}
		}
	}

	if uintptr(newcap) >= _MaxMem/uintptr(et.size) {
		panic(errorString("growslice: cap out of range"))
	}
	lenmem := uintptr(old.len) * uintptr(et.size)
	capmem := roundupsize(uintptr(newcap) * uintptr(et.size))
	newcap = int(capmem / uintptr(et.size))
	var p unsafe.Pointer
	if et.kind&kindNoPointers != 0 {
		p = rawmem(capmem)
		memmove(p, old.array, lenmem)
		memclr(add(p, lenmem), capmem-lenmem)
	} else {
		// Note: can't use rawmem (which avoids zeroing of memory), because then GC can scan unitialized memory.
		// TODO(rsc): Use memmove when !needwb().
		p = newarray(et, uintptr(newcap))
		for i := 0; i < old.len; i++ {
			typedmemmove(et, add(p, uintptr(i)*et.size), add(old.array, uintptr(i)*et.size))
		}
	}

	return slice{p, old.len, newcap}
}

func slicecopy(to, fm slice, width uintptr) int {
	if fm.len == 0 || to.len == 0 {
		return 0
	}

	n := fm.len
	if to.len < n {
		n = to.len
	}

	if width == 0 {
		return n
	}

	if raceenabled {
		callerpc := getcallerpc(unsafe.Pointer(&to))
		pc := funcPC(slicecopy)
		racewriterangepc(to.array, uintptr(n*int(width)), callerpc, pc)
		racereadrangepc(fm.array, uintptr(n*int(width)), callerpc, pc)
	}

	size := uintptr(n) * width
	if size == 1 { // common case worth about 2x to do here
		// TODO: is this still worth it with new memmove impl?
		*(*byte)(to.array) = *(*byte)(fm.array) // known to be a byte pointer
	} else {
		memmove(to.array, fm.array, size)
	}
	return int(n)
}

func slicestringcopy(to []byte, fm string) int {
	if len(fm) == 0 || len(to) == 0 {
		return 0
	}

	n := len(fm)
	if len(to) < n {
		n = len(to)
	}

	if raceenabled {
		callerpc := getcallerpc(unsafe.Pointer(&to))
		pc := funcPC(slicestringcopy)
		racewriterangepc(unsafe.Pointer(&to[0]), uintptr(n), callerpc, pc)
	}

	memmove(unsafe.Pointer(&to[0]), unsafe.Pointer((*stringStruct)(unsafe.Pointer(&fm)).str), uintptr(n))
	return n
}

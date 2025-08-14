// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scan

import (
	"internal/goarch"
	"internal/runtime/gc"
	"internal/runtime/sys"
	"unsafe"
)

// ScanSpanPackedGo is an optimized pure Go implementation of ScanSpanPacked.
func ScanSpanPackedGo(mem unsafe.Pointer, bufp *uintptr, objMarks *gc.ObjMask, sizeClass uintptr, ptrMask *gc.PtrMask) (count int32) {
	buf := newUnsafeBuf(bufp)
	objBytes := uintptr(gc.SizeClassToSize[sizeClass])
	// TODO(austin): Trim objMarks to the number of objects in this size class?
	for markI, markWord := range objMarks {
		for range sys.OnesCount64(uint64(markWord)) {
			bitI := sys.TrailingZeros64(uint64(markWord))
			markWord &^= 1 << bitI

			objIndex := markI*goarch.PtrBits + bitI

			// objStartInSpan is the index of the word from mem where the
			// object stats. objEndInSpan points to the next object, i.e.
			// it's an exclusive upper bound.
			objStartInSpan := objBytes * uintptr(objIndex) / goarch.PtrSize
			objEndInSpan := objStartInSpan + objBytes/goarch.PtrSize

			// TODO: Another way to do this would be to extract the pointer mask
			// for this object (it's at most 64 bits) and do a bit iteration
			// over that.

			for wordI := objStartInSpan; wordI < objEndInSpan; wordI++ {
				val := *(*uintptr)(unsafe.Add(mem, wordI*goarch.PtrSize))
				// Check if we should enqueue this word.
				//
				// We load the word before the check because, even though this
				// can lead to loading much more than necessary, it's faster.
				// Most likely this is because it warms up the hardware
				// prefetcher much better, and gives us more time before we need
				// the value.
				//
				// We discard values that can't possibly be useful pointers
				// here, too, because this filters out a lot of words and does
				// so with as little processing as possible.
				//
				// TODO: This is close to, but not entirely branchless.
				isPtr := bool2int(ptrMask[wordI/goarch.PtrBits]&(1<<(wordI%goarch.PtrBits)) != 0)
				isNonNil := bool2int(val >= 4096)
				pred := isPtr&isNonNil != 0
				buf.addIf(val, pred)
			}
		}
	}
	// We don't know the true size of bufp, but we can at least catch obvious errors
	// in this function by making sure we didn't write more than gc.PageWords pointers
	// into the buffer.
	buf.check(gc.PageWords)
	return int32(buf.n)
}

// unsafeBuf allows for appending to a buffer without bounds-checks or branches.
type unsafeBuf[T any] struct {
	base *T
	n    int
}

func newUnsafeBuf[T any](base *T) unsafeBuf[T] {
	return unsafeBuf[T]{base, 0}
}

// addIf appends a value to the buffer if the predicate is true.
//
// addIf speculatively writes to the next index of the buffer, so the caller
// must be certain that such a write will still be in-bounds with respect
// to the buffer's true capacity.
func (b *unsafeBuf[T]) addIf(val T, pred bool) {
	*(*T)(unsafe.Add(unsafe.Pointer(b.base), b.n*int(unsafe.Sizeof(val)))) = val
	b.n += bool2int(pred)
}

// check performs a bounds check on speculative writes into the buffer.
// Calling this shortly after a series of addIf calls is important to
// catch any misuse as fast as possible. Separating the bounds check from
// the append is more efficient, but one check to cover several appends is
// still efficient and much more memory safe.
func (b unsafeBuf[T]) check(cap int) {
	// We fail even if b.n == cap because addIf speculatively writes one past b.n.
	if b.n >= cap {
		panic("unsafeBuf overflow")
	}
}

func bool2int(x bool) int {
	// This particular pattern gets optimized by the compiler.
	var b int
	if x {
		b = 1
	}
	return b
}

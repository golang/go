// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/goarch"
	"internal/runtime/sys"
	"unsafe"
)

const (
	deferTryPtrs  = 4
	deferTryWords = 4
)

type _deferTry struct {
	d       _defer
	fv      uintptr
	errheap *error
	erroff  uintptr
	ext     *_deferTry
	ptrs    [deferTryPtrs]unsafe.Pointer
	words   [deferTryWords]uintptr
}

func deferprocTry(thunk uintptr, ctype *_type, clo unsafe.Pointer, roff uintptr) {
	gp := getg()
	if gp.m.curg != gp {
		throw("defer on system stack")
	}
	t := newdeferTry()
	packDeferTry(t, ctype, clo, roff)
	t.fv = thunk
	t.d.heap = true
	t.d.try = true
	fvp := unsafe.Pointer(&t.fv)
	t.d.fn = *(*func())(unsafe.Pointer(&fvp))
	t.d.link = gp._defer
	gp._defer = &t.d
	t.d.pc = sys.GetCallerPC()
	t.d.sp = sys.GetCallerSP()
}

func deferTryUnpack(rec unsafe.Pointer, ctype *_type, dst unsafe.Pointer, roff uintptr) {
	t := (*_deferTry)(unsafe.Pointer(uintptr(rec) - unsafe.Offsetof(_deferTry{}.fv)))
	unpackDeferTry(t, ctype, dst, roff)
	if t.errheap != nil {
		*(*unsafe.Pointer)(add(dst, roff)) = unsafe.Pointer(t.errheap)
	} else {
		*(*unsafe.Pointer)(add(dst, roff)) = unsafe.Pointer(getg().stack.hi - t.erroff)
	}
	freeDeferTry(t)
}

func packDeferTry(t *_deferTry, ctype *_type, clo unsafe.Pointer, roff uintptr) {
	errp := *(*unsafe.Pointer)(add(clo, roff))
	if getg().stack.lo <= uintptr(errp) && uintptr(errp) < getg().stack.hi {
		t.erroff = getg().stack.hi - uintptr(errp)
	} else {
		t.errheap = (*error)(errp)
	}
	n := ctype.Size_ / goarch.PtrSize
	np := ctype.PtrBytes / goarch.PtrSize
	mask := getGCMask(ctype)
	rec := t
	pi := 0
	wi := 0
	for i := uintptr(0); i < n; i++ {
		if i == roff/goarch.PtrSize {
			continue
		}
		if i < np && *(*byte)(add(unsafe.Pointer(mask), i/8))&(1<<(i%8)) != 0 {
			if pi == len(rec.ptrs) {
				rec.ext = newdeferTry()
				rec = rec.ext
				pi = 0
			}
			rec.ptrs[pi] = *(*unsafe.Pointer)(add(clo, i*goarch.PtrSize))
			pi++
			continue
		}
		if wi == len(rec.words) {
			rec.ext = newdeferTry()
			rec = rec.ext
			wi = 0
		}
		rec.words[wi] = *(*uintptr)(add(clo, i*goarch.PtrSize))
		wi++
	}
}

func unpackDeferTry(t *_deferTry, ctype *_type, dst unsafe.Pointer, roff uintptr) {
	n := ctype.Size_ / goarch.PtrSize
	np := ctype.PtrBytes / goarch.PtrSize
	mask := getGCMask(ctype)
	rec := t
	pi := 0
	wi := 0
	for i := uintptr(0); i < n; i++ {
		if i == roff/goarch.PtrSize {
			continue
		}
		if i < np && *(*byte)(add(unsafe.Pointer(mask), i/8))&(1<<(i%8)) != 0 {
			if pi == len(rec.ptrs) {
				rec = rec.ext
				pi = 0
			}
			*(*unsafe.Pointer)(add(dst, i*goarch.PtrSize)) = rec.ptrs[pi]
			pi++
			continue
		}
		if wi == len(rec.words) {
			rec = rec.ext
			wi = 0
		}
		*(*uintptr)(add(dst, i*goarch.PtrSize)) = rec.words[wi]
		wi++
	}
}

func newdeferTry() *_deferTry {
	var t *_deferTry
	mp := acquirem()
	pp := mp.p.ptr()
	if len(pp.deferTryPool) == 0 && sched.deferTryPool != nil {
		lock(&sched.deferlock)
		for len(pp.deferTryPool) < cap(pp.deferTryPool)/2 && sched.deferTryPool != nil {
			r := sched.deferTryPool
			sched.deferTryPool = r.ext
			r.ext = nil
			pp.deferTryPool = append(pp.deferTryPool, r)
		}
		unlock(&sched.deferlock)
	}
	if n := len(pp.deferTryPool); n > 0 {
		t = pp.deferTryPool[n-1]
		pp.deferTryPool[n-1] = nil
		pp.deferTryPool = pp.deferTryPool[:n-1]
	}
	releasem(mp)
	mp, pp = nil, nil
	if t == nil {
		t = new(_deferTry)
	}
	return t
}

func freeDeferTry(t *_deferTry) {
	for t != nil {
		n := t.ext
		*t = _deferTry{}
		mp := acquirem()
		pp := mp.p.ptr()
		if len(pp.deferTryPool) == cap(pp.deferTryPool) {
			var first, last *_deferTry
			for len(pp.deferTryPool) > cap(pp.deferTryPool)/2 {
				m := len(pp.deferTryPool)
				r := pp.deferTryPool[m-1]
				pp.deferTryPool[m-1] = nil
				pp.deferTryPool = pp.deferTryPool[:m-1]
				if first == nil {
					first = r
				} else {
					last.ext = r
				}
				last = r
			}
			lock(&sched.deferlock)
			last.ext = sched.deferTryPool
			sched.deferTryPool = first
			unlock(&sched.deferlock)
		}
		pp.deferTryPool = append(pp.deferTryPool, t)
		releasem(mp)
		t = n
	}
}

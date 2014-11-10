// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// Called from C. Returns the Go type *m.
func gc_m_ptr(ret *interface{}) {
	*ret = (*m)(nil)
}

// Called from C. Returns the Go type *g.
func gc_g_ptr(ret *interface{}) {
	*ret = (*g)(nil)
}

// Called from C. Returns the Go type *itab.
func gc_itab_ptr(ret *interface{}) {
	*ret = (*itab)(nil)
}

func gc_unixnanotime(now *int64) {
	sec, nsec := timenow()
	*now = sec*1e9 + int64(nsec)
}

func freeOSMemory() {
	gogc(2) // force GC and do eager sweep
	onM(scavenge_m)
}

var poolcleanup func()

func registerPoolCleanup(f func()) {
	poolcleanup = f
}

func clearpools() {
	// clear sync.Pools
	if poolcleanup != nil {
		poolcleanup()
	}

	for _, p := range &allp {
		if p == nil {
			break
		}
		// clear tinyalloc pool
		if c := p.mcache; c != nil {
			c.tiny = nil
			c.tinysize = 0
			c.sudogcache = nil
		}
		// clear defer pools
		for i := range p.deferpool {
			p.deferpool[i] = nil
		}
	}
}

func gosweepone() uintptr
func gosweepdone() bool

func bgsweep() {
	getg().issystem = true
	for {
		for gosweepone() != ^uintptr(0) {
			sweep.nbgsweep++
			Gosched()
		}
		lock(&gclock)
		if !gosweepdone() {
			// This can happen if a GC runs between
			// gosweepone returning ^0 above
			// and the lock being acquired.
			unlock(&gclock)
			continue
		}
		sweep.parked = true
		goparkunlock(&gclock, "GC sweep wait")
	}
}

const (
	_PoisonGC    = 0xf969696969696969 & (1<<(8*ptrSize) - 1)
	_PoisonStack = 0x6868686868686868 & (1<<(8*ptrSize) - 1)
)

// NOTE: Really dst *unsafe.Pointer, src unsafe.Pointer,
// but if we do that, Go inserts a write barrier on *dst = src.
//go:nosplit
func writebarrierptr(dst *uintptr, src uintptr) {
	*dst = src
	writebarrierptr_nostore(dst, src)
}

// Like writebarrierptr, but the store has already been applied.
// Do not reapply.
//go:nosplit
func writebarrierptr_nostore(dst *uintptr, src uintptr) {
	if getg() == nil { // very low-level startup
		return
	}

	if src != 0 && (src < _PageSize || src == _PoisonGC || src == _PoisonStack) {
		onM(func() { gothrow("bad pointer in write barrier") })
	}

	mp := acquirem()
	if mp.inwb {
		releasem(mp)
		return
	}
	mp.inwb = true
	oldscalar0 := mp.scalararg[0]
	oldscalar1 := mp.scalararg[1]
	mp.scalararg[0] = uintptr(unsafe.Pointer(dst))
	mp.scalararg[1] = src
	onM_signalok(gcmarkwb_m)
	mp.scalararg[0] = oldscalar0
	mp.scalararg[1] = oldscalar1
	mp.inwb = false
	releasem(mp)
}

//go:nosplit
func writebarrierstring(dst *[2]uintptr, src [2]uintptr) {
	writebarrierptr(&dst[0], src[0])
	dst[1] = src[1]
}

//go:nosplit
func writebarrierslice(dst *[3]uintptr, src [3]uintptr) {
	writebarrierptr(&dst[0], src[0])
	dst[1] = src[1]
	dst[2] = src[2]
}

//go:nosplit
func writebarrieriface(dst *[2]uintptr, src [2]uintptr) {
	writebarrierptr(&dst[0], src[0])
	writebarrierptr(&dst[1], src[1])
}

//go:generate go run wbfat_gen.go -- wbfat.go
//
// The above line generates multiword write barriers for
// all the combinations of ptr+scalar up to four words.
// The implementations are written to wbfat.go.

//go:nosplit
func writebarrierfat(typ *_type, dst, src unsafe.Pointer) {
	mask := loadPtrMask(typ)
	nptr := typ.size / ptrSize
	for i := uintptr(0); i < nptr; i += 2 {
		bits := mask[i/2]
		if (bits>>2)&_BitsMask == _BitsPointer {
			writebarrierptr((*uintptr)(dst), *(*uintptr)(src))
		} else {
			*(*uintptr)(dst) = *(*uintptr)(src)
		}
		dst = add(dst, ptrSize)
		src = add(src, ptrSize)
		if i+1 == nptr {
			break
		}
		bits >>= 4
		if (bits>>2)&_BitsMask == _BitsPointer {
			writebarrierptr((*uintptr)(dst), *(*uintptr)(src))
		} else {
			*(*uintptr)(dst) = *(*uintptr)(src)
		}
		dst = add(dst, ptrSize)
		src = add(src, ptrSize)
	}
}

//go:nosplit
func writebarriercopy(typ *_type, dst, src slice) int {
	n := dst.len
	if n > src.len {
		n = src.len
	}
	if n == 0 {
		return 0
	}
	dstp := unsafe.Pointer(dst.array)
	srcp := unsafe.Pointer(src.array)

	if uintptr(srcp) < uintptr(dstp) && uintptr(srcp)+uintptr(n)*typ.size > uintptr(dstp) {
		// Overlap with src before dst.
		// Copy backward, being careful not to move dstp/srcp
		// out of the array they point into.
		dstp = add(dstp, uintptr(n-1)*typ.size)
		srcp = add(srcp, uintptr(n-1)*typ.size)
		i := uint(0)
		for {
			writebarrierfat(typ, dstp, srcp)
			if i++; i >= n {
				break
			}
			dstp = add(dstp, -typ.size)
			srcp = add(srcp, -typ.size)
		}
	} else {
		// Copy forward, being careful not to move dstp/srcp
		// out of the array they point into.
		i := uint(0)
		for {
			writebarrierfat(typ, dstp, srcp)
			if i++; i >= n {
				break
			}
			dstp = add(dstp, typ.size)
			srcp = add(srcp, typ.size)
		}
	}
	return int(n)
}

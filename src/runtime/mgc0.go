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

			// disconnect cached list before dropping it on the floor,
			// so that a dangling ref to one entry does not pin all of them.
			var sg, sgnext *sudog
			for sg = c.sudogcache; sg != nil; sg = sgnext {
				sgnext = sg.next
				sg.next = nil
			}
			c.sudogcache = nil
		}

		// clear defer pools
		for i := range p.deferpool {
			// disconnect cached list before dropping it on the floor,
			// so that a dangling ref to one entry does not pin all of them.
			var d, dlink *_defer
			for d = p.deferpool[i]; d != nil; d = dlink {
				dlink = d.link
				d.link = nil
			}
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

// NOTE: Really dst *unsafe.Pointer, src unsafe.Pointer,
// but if we do that, Go inserts a write barrier on *dst = src.
//go:nosplit
func writebarrierptr(dst *uintptr, src uintptr) {
	*dst = src
}

//go:nosplit
func writebarrierstring(dst *[2]uintptr, src [2]uintptr) {
	dst[0] = src[0]
	dst[1] = src[1]
}

//go:nosplit
func writebarrierslice(dst *[3]uintptr, src [3]uintptr) {
	dst[0] = src[0]
	dst[1] = src[1]
	dst[2] = src[2]
}

//go:nosplit
func writebarrieriface(dst *[2]uintptr, src [2]uintptr) {
	dst[0] = src[0]
	dst[1] = src[1]
}

//go:nosplit
func writebarrierfat2(dst *[2]uintptr, _ *byte, src [2]uintptr) {
	dst[0] = src[0]
	dst[1] = src[1]
}

//go:nosplit
func writebarrierfat3(dst *[3]uintptr, _ *byte, src [3]uintptr) {
	dst[0] = src[0]
	dst[1] = src[1]
	dst[2] = src[2]
}

//go:nosplit
func writebarrierfat4(dst *[4]uintptr, _ *byte, src [4]uintptr) {
	dst[0] = src[0]
	dst[1] = src[1]
	dst[2] = src[2]
	dst[3] = src[3]
}

//go:nosplit
func writebarrierfat(typ *_type, dst, src unsafe.Pointer) {
	memmove(dst, src, typ.size)
}

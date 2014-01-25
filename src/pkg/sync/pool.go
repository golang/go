// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync

import (
	"runtime"
	"sync/atomic"
	"unsafe"
)

const (
	cacheLineSize = 128
	poolLocalSize = 2 * cacheLineSize
	poolLocalCap  = poolLocalSize/unsafe.Sizeof(*(*interface{})(nil)) - 1
)

// A Pool is a set of temporary objects that may be individually saved
// and retrieved.
//
// Any item stored in the Pool may be removed automatically by the
// implementation at any time without notification.
// If the Pool holds the only reference when this happens, the item
// might be deallocated.
//
// A Pool is safe for use by multiple goroutines simultaneously.
//
// Pool's intended use is for free lists maintained in global variables,
// typically accessed by multiple goroutines simultaneously. Using a
// Pool instead of a custom free list allows the runtime to reclaim
// entries from the pool when it makes sense to do so. An
// appropriate use of sync.Pool is to create a pool of temporary buffers
// shared between independent clients of a global resource. On the
// other hand, if a free list is maintained as part of an object used
// only by a single client and freed when the client completes,
// implementing that free list as a Pool is not appropriate.
//
// This is an experimental type and might not be released.
type Pool struct {
	// The following fields are known to runtime.
	next         *Pool      // for use by runtime
	local        *poolLocal // local fixed-size per-P pool, actually an array
	localSize    uintptr    // size of the local array
	globalOffset uintptr    // offset of global
	// The rest is not known to runtime.

	// New optionally specifies a function to generate
	// a value when Get would otherwise return nil.
	// It may not be changed concurrently with calls to Get.
	New func() interface{}

	pad [cacheLineSize]byte
	// Read-mostly date above this point, mutable data follows.
	mu     Mutex
	global []interface{} // global fallback pool
}

// Local per-P Pool appendix.
type poolLocal struct {
	tail   int
	unused int
	buf    [poolLocalCap]interface{}
}

func init() {
	var v poolLocal
	if unsafe.Sizeof(v) != poolLocalSize {
		panic("sync: incorrect pool size")
	}
}

// Put adds x to the pool.
func (p *Pool) Put(x interface{}) {
	if raceenabled {
		// Under race detector the Pool degenerates into no-op.
		// It's conforming, simple and does not introduce excessive
		// happens-before edges between unrelated goroutines.
		return
	}
	if x == nil {
		return
	}
	l := p.pin()
	t := l.tail
	if t < int(poolLocalCap) {
		l.buf[t] = x
		l.tail = t + 1
		runtime_procUnpin()
		return
	}
	p.putSlow(l, x)
}

// Get selects an arbitrary item from the Pool, removes it from the
// Pool, and returns it to the caller.
// Get may choose to ignore the pool and treat it as empty.
// Callers should not assume any relation between values passed to Put and
// the values returned by Get.
//
// If Get would otherwise return nil and p.New is non-nil, Get returns
// the result of calling p.New.
func (p *Pool) Get() interface{} {
	if raceenabled {
		if p.New != nil {
			return p.New()
		}
		return nil
	}
	l := p.pin()
	t := l.tail
	if t > 0 {
		t -= 1
		x := l.buf[t]
		l.tail = t
		runtime_procUnpin()
		return x
	}
	return p.getSlow()
}

func (p *Pool) putSlow(l *poolLocal, x interface{}) {
	// Grab half of items from local pool and put to global pool.
	// Can not lock the mutex while pinned.
	const N = int(poolLocalCap/2 + 1)
	var buf [N]interface{}
	buf[0] = x
	for i := 1; i < N; i++ {
		l.tail--
		buf[i] = l.buf[l.tail]
	}
	runtime_procUnpin()

	p.mu.Lock()
	p.global = append(p.global, buf[:]...)
	p.mu.Unlock()
}

func (p *Pool) getSlow() (x interface{}) {
	// Grab a batch of items from global pool and put to local pool.
	// Can not lock the mutex while pinned.
	runtime_procUnpin()
	p.mu.Lock()
	pid := runtime_procPin()
	s := p.localSize
	l := p.local
	if uintptr(pid) < s {
		l = indexLocal(l, pid)
		// Get the item to return.
		last := len(p.global) - 1
		if last >= 0 {
			x = p.global[last]
			p.global = p.global[:last]
		}
		// Try to refill local pool, we may have been rescheduled to another P.
		if last > 0 && l.tail == 0 {
			n := int(poolLocalCap / 2)
			gl := len(p.global)
			if n > gl {
				n = gl
			}
			copy(l.buf[:], p.global[gl-n:])
			p.global = p.global[:gl-n]
			l.tail = n
		}
	}
	runtime_procUnpin()
	p.mu.Unlock()

	if x == nil && p.New != nil {
		x = p.New()
	}
	return
}

// pin pins current goroutine to P, disables preemption and returns poolLocal pool for the P.
// Caller must call runtime_procUnpin() when done with the pool.
func (p *Pool) pin() *poolLocal {
	pid := runtime_procPin()
	// In pinSlow we store to localSize and then to local, here we load in opposite order.
	// Since we've disabled preemption, GC can not happen in between.
	// Thus here we must observe local at least as large localSize.
	// We can observe a newer/larger local, it is fine (we must observe its zero-initialized-ness).
	s := atomic.LoadUintptr(&p.localSize) // load-acquire
	l := p.local                          // load-consume
	if uintptr(pid) < s {
		return indexLocal(l, pid)
	}
	return p.pinSlow()
}

func (p *Pool) pinSlow() *poolLocal {
	// Retry under the mutex.
	runtime_procUnpin()
	p.mu.Lock()
	defer p.mu.Unlock()
	pid := runtime_procPin()
	s := p.localSize
	l := p.local
	if uintptr(pid) < s {
		return indexLocal(l, pid)
	}
	if p.local == nil {
		p.globalOffset = unsafe.Offsetof(p.global)
		runtime_registerPool(p)
	}
	// If GOMAXPROCS changes between GCs, we re-allocate the array and lose the old one.
	size := runtime.GOMAXPROCS(0)
	local := make([]poolLocal, size)
	atomic.StorePointer((*unsafe.Pointer)(unsafe.Pointer(&p.local)), unsafe.Pointer(&local[0])) // store-release
	atomic.StoreUintptr(&p.localSize, uintptr(size))                                            // store-release
	return &local[pid]
}

func indexLocal(l *poolLocal, i int) *poolLocal {
	return (*poolLocal)(unsafe.Pointer(uintptr(unsafe.Pointer(l)) + unsafe.Sizeof(*l)*uintptr(i))) // uh...
}

// Implemented in runtime.
func runtime_registerPool(*Pool)
func runtime_procPin() int
func runtime_procUnpin()

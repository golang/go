// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Semaphore implementation exposed to Go.
// Intended use is provide a sleep and wakeup
// primitive that can be used in the contended case
// of other synchronization primitives.
// Thus it targets the same goal as Linux's futex,
// but it has much simpler semantics.
//
// That is, don't think of these as semaphores.
// Think of them as a way to implement sleep and wakeup
// such that every sleep is paired with a single wakeup,
// even if, due to races, the wakeup happens before the sleep.
//
// See Mullender and Cox, ``Semaphores in Plan 9,''
// http://swtch.com/semaphore.pdf

package runtime

import "unsafe"

// Synchronous semaphore for sync.Cond.
type syncSema struct {
	lock lock
	head *sudog
	tail *sudog
}

// Syncsemacquire waits for a pairing syncsemrelease on the same semaphore s.
func syncsemacquire(s *syncSema) {
	golock(&s.lock)
	if s.head != nil && s.head.nrelease > 0 {
		// Have pending release, consume it.
		var wake *sudog
		s.head.nrelease--
		if s.head.nrelease == 0 {
			wake = s.head
			s.head = wake.link
			if s.head == nil {
				s.tail = nil
			}
		}
		gounlock(&s.lock)
		if wake != nil {
			goready(wake.g)
		}
	} else {
		// Enqueue itself.
		w := acquireSudog()
		w.g = getg()
		w.nrelease = -1
		w.link = nil
		w.releasetime = 0
		t0 := int64(0)
		if blockprofilerate > 0 {
			t0 = gocputicks()
			w.releasetime = -1
		}
		if s.tail == nil {
			s.head = w
		} else {
			s.tail.link = w
		}
		s.tail = w
		goparkunlock(&s.lock, "semacquire")
		if t0 != 0 {
			goblockevent(int64(w.releasetime)-t0, 3)
		}
		releaseSudog(w)
	}
}

// Syncsemrelease waits for n pairing syncsemacquire on the same semaphore s.
func syncsemrelease(s *syncSema, n uint32) {
	golock(&s.lock)
	for n > 0 && s.head != nil && s.head.nrelease < 0 {
		// Have pending acquire, satisfy it.
		wake := s.head
		s.head = wake.link
		if s.head == nil {
			s.tail = nil
		}
		if wake.releasetime != 0 {
			// TODO: Remove use of unsafe here.
			releasetimep := (*int64)(unsafe.Pointer(&wake.releasetime))
			*releasetimep = gocputicks()
		}
		goready(wake.g)
		n--
	}
	if n > 0 {
		// enqueue itself
		w := acquireSudog()
		w.g = getg()
		w.nrelease = int32(n)
		w.link = nil
		w.releasetime = 0
		if s.tail == nil {
			s.head = w
		} else {
			s.tail.link = w
		}
		s.tail = w
		goparkunlock(&s.lock, "semarelease")
	} else {
		gounlock(&s.lock)
	}
}

func syncsemcheck(sz uintptr) {
	if sz != unsafe.Sizeof(syncSema{}) {
		print("runtime: bad syncSema size - sync=", sz, " runtime=", unsafe.Sizeof(syncSema{}), "\n")
		gothrow("bad syncSema size")
	}
}

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || netbsd || openbsd

package runtime

// Integrated network poller (kqueue-based implementation).

import (
	"internal/goarch"
	"internal/runtime/atomic"
	"unsafe"
)

var (
	kq             int32         = -1
	netpollWakeSig atomic.Uint32 // used to avoid duplicate calls of netpollBreak
)

func netpollinit() {
	kq = kqueue()
	if kq < 0 {
		println("runtime: kqueue failed with", -kq)
		throw("runtime: netpollinit failed")
	}
	closeonexec(kq)
	addWakeupEvent(kq)
}

func netpollopen(fd uintptr, pd *pollDesc) int32 {
	// Arm both EVFILT_READ and EVFILT_WRITE in edge-triggered mode (EV_CLEAR)
	// for the whole fd lifetime. The notifications are automatically unregistered
	// when fd is closed.
	var ev [2]keventt
	*(*uintptr)(unsafe.Pointer(&ev[0].ident)) = fd
	ev[0].filter = _EVFILT_READ
	ev[0].flags = _EV_ADD | _EV_CLEAR
	ev[0].fflags = 0
	ev[0].data = 0

	if goarch.PtrSize == 4 {
		// We only have a pointer-sized field to store into,
		// so on a 32-bit system we get no sequence protection.
		// TODO(iant): If we notice any problems we could at least
		// steal the low-order 2 bits for a tiny sequence number.
		ev[0].udata = (*byte)(unsafe.Pointer(pd))
	} else {
		tp := taggedPointerPack(unsafe.Pointer(pd), pd.fdseq.Load())
		ev[0].udata = (*byte)(unsafe.Pointer(uintptr(tp)))
	}
	ev[1] = ev[0]
	ev[1].filter = _EVFILT_WRITE
	n := kevent(kq, &ev[0], 2, nil, 0, nil)
	if n < 0 {
		return -n
	}
	return 0
}

func netpollclose(fd uintptr) int32 {
	// Don't need to unregister because calling close()
	// on fd will remove any kevents that reference the descriptor.
	return 0
}

func netpollarm(pd *pollDesc, mode int) {
	throw("runtime: unused")
}

// netpollBreak interrupts a kevent.
func netpollBreak() {
	// Failing to cas indicates there is an in-flight wakeup, so we're done here.
	if !netpollWakeSig.CompareAndSwap(0, 1) {
		return
	}

	wakeNetpoll(kq)
}

// netpoll checks for ready network connections.
// Returns a list of goroutines that become runnable,
// and a delta to add to netpollWaiters.
// This must never return an empty list with a non-zero delta.
//
// delay < 0: blocks indefinitely
// delay == 0: does not block, just polls
// delay > 0: block for up to that many nanoseconds
func netpoll(delay int64) (gList, int32) {
	if kq == -1 {
		return gList{}, 0
	}
	var tp *timespec
	var ts timespec
	if delay < 0 {
		tp = nil
	} else if delay == 0 {
		tp = &ts
	} else {
		ts.setNsec(delay)
		if ts.tv_sec > 1e6 {
			// Darwin returns EINVAL if the sleep time is too long.
			ts.tv_sec = 1e6
		}
		tp = &ts
	}
	var events [64]keventt
retry:
	n := kevent(kq, nil, 0, &events[0], int32(len(events)), tp)
	if n < 0 {
		// Ignore the ETIMEDOUT error for now, but try to dive deep and
		// figure out what really happened with n == ETIMEOUT,
		// see https://go.dev/issue/59679 for details.
		if n != -_EINTR && n != -_ETIMEDOUT {
			println("runtime: kevent on fd", kq, "failed with", -n)
			throw("runtime: netpoll failed")
		}
		// If a timed sleep was interrupted, just return to
		// recalculate how long we should sleep now.
		if delay > 0 {
			return gList{}, 0
		}
		goto retry
	}
	var toRun gList
	delta := int32(0)
	for i := 0; i < int(n); i++ {
		ev := &events[i]

		if isWakeup(ev) {
			isBlocking := delay != 0
			processWakeupEvent(kq, isBlocking)
			if isBlocking {
				// netpollBreak could be picked up by a nonblocking poll.
				// Only reset the netpollWakeSig if blocking.
				netpollWakeSig.Store(0)
			}
			continue
		}

		var mode int32
		switch ev.filter {
		case _EVFILT_READ:
			mode += 'r'

			// On some systems when the read end of a pipe
			// is closed the write end will not get a
			// _EVFILT_WRITE event, but will get a
			// _EVFILT_READ event with EV_EOF set.
			// Note that setting 'w' here just means that we
			// will wake up a goroutine waiting to write;
			// that goroutine will try the write again,
			// and the appropriate thing will happen based
			// on what that write returns (success, EPIPE, EAGAIN).
			if ev.flags&_EV_EOF != 0 {
				mode += 'w'
			}
		case _EVFILT_WRITE:
			mode += 'w'
		}
		if mode != 0 {
			var pd *pollDesc
			var tag uintptr
			if goarch.PtrSize == 4 {
				// No sequence protection on 32-bit systems.
				// See netpollopen for details.
				pd = (*pollDesc)(unsafe.Pointer(ev.udata))
				tag = 0
			} else {
				tp := taggedPointer(uintptr(unsafe.Pointer(ev.udata)))
				pd = (*pollDesc)(tp.pointer())
				tag = tp.tag()
				if pd.fdseq.Load() != tag {
					continue
				}
			}
			pd.setEventErr(ev.flags == _EV_ERROR, tag)
			delta += netpollready(&toRun, pd, mode)
		}
	}
	return toRun, delta
}

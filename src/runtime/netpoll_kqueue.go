// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd netbsd openbsd

package runtime

// Integrated network poller (kqueue-based implementation).

import "unsafe"

var (
	kq int32 = -1

	netpollBreakRd, netpollBreakWr uintptr // for netpollBreak
)

func netpollinit() {
	kq = kqueue()
	if kq < 0 {
		println("runtime: kqueue failed with", -kq)
		throw("runtime: netpollinit failed")
	}
	closeonexec(kq)
	r, w, errno := nonblockingPipe()
	if errno != 0 {
		println("runtime: pipe failed with", -errno)
		throw("runtime: pipe failed")
	}
	ev := keventt{
		filter: _EVFILT_READ,
		flags:  _EV_ADD,
	}
	*(*uintptr)(unsafe.Pointer(&ev.ident)) = uintptr(r)
	n := kevent(kq, &ev, 1, nil, 0, nil)
	if n < 0 {
		println("runtime: kevent failed with", -n)
		throw("runtime: kevent failed")
	}
	netpollBreakRd = uintptr(r)
	netpollBreakWr = uintptr(w)
}

func netpollIsPollDescriptor(fd uintptr) bool {
	return fd == uintptr(kq) || fd == netpollBreakRd || fd == netpollBreakWr
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
	ev[0].udata = (*byte)(unsafe.Pointer(pd))
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

// netpollBreak interrupts an epollwait.
func netpollBreak() {
	for {
		var b byte
		n := write(netpollBreakWr, unsafe.Pointer(&b), 1)
		if n == 1 || n == -_EAGAIN {
			break
		}
		if n == -_EINTR {
			continue
		}
		println("runtime: netpollBreak write failed with", -n)
		throw("runtime: netpollBreak write failed")
	}
}

// netpoll checks for ready network connections.
// Returns list of goroutines that become runnable.
// delay < 0: blocks indefinitely
// delay == 0: does not block, just polls
// delay > 0: block for up to that many nanoseconds
func netpoll(delay int64) gList {
	if kq == -1 {
		return gList{}
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
		if n != -_EINTR {
			println("runtime: kevent on fd", kq, "failed with", -n)
			throw("runtime: netpoll failed")
		}
		// If a timed sleep was interrupted, just return to
		// recalculate how long we should sleep now.
		if delay > 0 {
			return gList{}
		}
		goto retry
	}
	var toRun gList
	for i := 0; i < int(n); i++ {
		ev := &events[i]

		if uintptr(ev.ident) == netpollBreakRd {
			if ev.filter != _EVFILT_READ {
				println("runtime: netpoll: break fd ready for", ev.filter)
				throw("runtime: netpoll: break fd ready for something unexpected")
			}
			if delay != 0 {
				// netpollBreak could be picked up by a
				// nonblocking poll. Only read the byte
				// if blocking.
				var tmp [16]byte
				read(int32(netpollBreakRd), noescape(unsafe.Pointer(&tmp[0])), int32(len(tmp)))
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
			pd := (*pollDesc)(unsafe.Pointer(ev.udata))
			pd.everr = false
			if ev.flags == _EV_ERROR {
				pd.everr = true
			}
			netpollready(&toRun, pd, mode)
		}
	}
	return toRun
}

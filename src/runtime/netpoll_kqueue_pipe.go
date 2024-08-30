// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netbsd || openbsd

package runtime

import "unsafe"

// TODO(panjf2000): NetBSD didn't implement EVFILT_USER for user-established events
// until NetBSD 10.0, check out https://www.netbsd.org/releases/formal-10/NetBSD-10.0.html
// Therefore we use the pipe to wake up the kevent on NetBSD at this point. Get back here
// and switch to EVFILT_USER when we bump up the minimal requirement of NetBSD to 10.0.
// Alternatively, maybe we can use EVFILT_USER on the NetBSD by checking the kernel version
// via uname(3) and fall back to the pipe if the kernel version is older than 10.0.

var netpollBreakRd, netpollBreakWr uintptr // for netpollBreak

func addWakeupEvent(kq int32) {
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

func wakeNetpoll(_ int32) {
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

func isWakeup(ev *keventt) bool {
	if uintptr(ev.ident) == netpollBreakRd {
		if ev.filter == _EVFILT_READ {
			return true
		}
		println("runtime: netpoll: break fd ready for", ev.filter)
		throw("runtime: netpoll: break fd ready for something unexpected")
	}
	return false
}

func drainWakeupEvent(_ int32) {
	var buf [16]byte
	read(int32(netpollBreakRd), noescape(unsafe.Pointer(&buf[0])), int32(len(buf)))
}

func netpollIsPollDescriptor(fd uintptr) bool {
	return fd == uintptr(kq) || fd == netpollBreakRd || fd == netpollBreakWr
}

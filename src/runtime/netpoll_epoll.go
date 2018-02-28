// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux

package runtime

import "unsafe"

func epollcreate(size int32) int32
func epollcreate1(flags int32) int32

//go:noescape
func epollctl(epfd, op, fd int32, ev *epollevent) int32

//go:noescape
func epollwait(epfd int32, ev *epollevent, nev, timeout int32) int32
func closeonexec(fd int32)

var (
	epfd int32 = -1 // epoll descriptor
)

func netpollinit() {
	epfd = epollcreate1(_EPOLL_CLOEXEC)
	if epfd >= 0 {
		return
	}
	epfd = epollcreate(1024)
	if epfd >= 0 {
		closeonexec(epfd)
		return
	}
	println("runtime: epollcreate failed with", -epfd)
	throw("runtime: netpollinit failed")
}

func netpolldescriptor() uintptr {
	return uintptr(epfd)
}

func netpollopen(fd uintptr, pd *pollDesc) int32 {
	var ev epollevent
	ev.events = _EPOLLIN | _EPOLLOUT | _EPOLLRDHUP | _EPOLLET
	*(**pollDesc)(unsafe.Pointer(&ev.data)) = pd
	return -epollctl(epfd, _EPOLL_CTL_ADD, int32(fd), &ev)
}

func netpollclose(fd uintptr) int32 {
	var ev epollevent
	return -epollctl(epfd, _EPOLL_CTL_DEL, int32(fd), &ev)
}

func netpollarm(pd *pollDesc, mode int) {
	throw("runtime: unused")
}

// polls for ready network connections
// returns list of goroutines that become runnable
func netpoll(block bool) *g {
	if epfd == -1 {
		return nil
	}
	waitms := int32(-1)
	if !block {
		waitms = 0
	}
	var events [128]epollevent
retry:
	n := epollwait(epfd, &events[0], int32(len(events)), waitms)
	if n < 0 {
		if n != -_EINTR {
			println("runtime: epollwait on fd", epfd, "failed with", -n)
			throw("runtime: netpoll failed")
		}
		goto retry
	}
	var gp guintptr
	for i := int32(0); i < n; i++ {
		ev := &events[i]
		if ev.events == 0 {
			continue
		}
		var mode int32
		if ev.events&(_EPOLLIN|_EPOLLRDHUP|_EPOLLHUP|_EPOLLERR) != 0 {
			mode += 'r'
		}
		if ev.events&(_EPOLLOUT|_EPOLLHUP|_EPOLLERR) != 0 {
			mode += 'w'
		}
		if mode != 0 {
			pd := *(**pollDesc)(unsafe.Pointer(&ev.data))

			netpollready(&gp, pd, mode)
		}
	}
	if block && gp == 0 {
		goto retry
	}
	return gp.ptr()
}

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Waiting for FDs via epoll(7).

package net

import (
	"os"
	"syscall"
)

const (
	readFlags  = syscall.EPOLLIN | syscall.EPOLLRDHUP
	writeFlags = syscall.EPOLLOUT
)

type pollster struct {
	epfd int

	// Events we're already waiting for
	events map[int]uint32
}

func newpollster() (p *pollster, err os.Error) {
	p = new(pollster)
	var e int

	// The arg to epoll_create is a hint to the kernel
	// about the number of FDs we will care about.
	// We don't know.
	if p.epfd, e = syscall.EpollCreate(16); e != 0 {
		return nil, os.NewSyscallError("epoll_create", e)
	}
	p.events = make(map[int]uint32)
	return p, nil
}

func (p *pollster) AddFD(fd int, mode int, repeat bool) os.Error {
	var ev syscall.EpollEvent
	var already bool
	ev.Fd = int32(fd)
	ev.Events, already = p.events[fd]
	if !repeat {
		ev.Events |= syscall.EPOLLONESHOT
	}
	if mode == 'r' {
		ev.Events |= readFlags
	} else {
		ev.Events |= writeFlags
	}

	var op int
	if already {
		op = syscall.EPOLL_CTL_MOD
	} else {
		op = syscall.EPOLL_CTL_ADD
	}
	if e := syscall.EpollCtl(p.epfd, op, fd, &ev); e != 0 {
		return os.NewSyscallError("epoll_ctl", e)
	}
	p.events[fd] = ev.Events
	return nil
}

func (p *pollster) StopWaiting(fd int, bits uint) {
	events, already := p.events[fd]
	if !already {
		print("Epoll unexpected fd=", fd, "\n")
		return
	}

	// If syscall.EPOLLONESHOT is not set, the wait
	// is a repeating wait, so don't change it.
	if events&syscall.EPOLLONESHOT == 0 {
		return
	}

	// Disable the given bits.
	// If we're still waiting for other events, modify the fd
	// event in the kernel.  Otherwise, delete it.
	events &= ^uint32(bits)
	if int32(events)&^syscall.EPOLLONESHOT != 0 {
		var ev syscall.EpollEvent
		ev.Fd = int32(fd)
		ev.Events = events
		if e := syscall.EpollCtl(p.epfd, syscall.EPOLL_CTL_MOD, fd, &ev); e != 0 {
			print("Epoll modify fd=", fd, ": ", os.Errno(e).String(), "\n")
		}
		p.events[fd] = events
	} else {
		if e := syscall.EpollCtl(p.epfd, syscall.EPOLL_CTL_DEL, fd, nil); e != 0 {
			print("Epoll delete fd=", fd, ": ", os.Errno(e).String(), "\n")
		}
		p.events[fd] = 0, false
	}
}

func (p *pollster) DelFD(fd int, mode int) {
	if mode == 'r' {
		p.StopWaiting(fd, readFlags)
	} else {
		p.StopWaiting(fd, writeFlags)
	}
}

func (p *pollster) WaitFD(nsec int64) (fd int, mode int, err os.Error) {
	// Get an event.
	var evarray [1]syscall.EpollEvent
	ev := &evarray[0]
	var msec int = -1
	if nsec > 0 {
		msec = int((nsec + 1e6 - 1) / 1e6)
	}
	n, e := syscall.EpollWait(p.epfd, evarray[0:], msec)
	for e == syscall.EAGAIN || e == syscall.EINTR {
		n, e = syscall.EpollWait(p.epfd, evarray[0:], msec)
	}
	if e != 0 {
		return -1, 0, os.NewSyscallError("epoll_wait", e)
	}
	if n == 0 {
		return -1, 0, nil
	}
	fd = int(ev.Fd)

	if ev.Events&writeFlags != 0 {
		p.StopWaiting(fd, writeFlags)
		return fd, 'w', nil
	}
	if ev.Events&readFlags != 0 {
		p.StopWaiting(fd, readFlags)
		return fd, 'r', nil
	}

	// Other events are error conditions - wake whoever is waiting.
	events, _ := p.events[fd]
	if events&writeFlags != 0 {
		p.StopWaiting(fd, writeFlags)
		return fd, 'w', nil
	}
	p.StopWaiting(fd, readFlags)
	return fd, 'r', nil
}

func (p *pollster) Close() os.Error {
	return os.NewSyscallError("close", syscall.Close(p.epfd))
}

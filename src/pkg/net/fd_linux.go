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
	// Must hold pollServer lock
	events map[int]uint32

	// An event buffer for EpollWait.
	// Used without a lock, may only be used by WaitFD.
	waitEventBuf [10]syscall.EpollEvent
	waitEvents   []syscall.EpollEvent

	// An event buffer for EpollCtl, to avoid a malloc.
	// Must hold pollServer lock.
	ctlEvent syscall.EpollEvent
}

func newpollster() (p *pollster, err error) {
	p = new(pollster)
	if p.epfd, err = syscall.EpollCreate1(syscall.EPOLL_CLOEXEC); err != nil {
		if err != syscall.ENOSYS {
			return nil, os.NewSyscallError("epoll_create1", err)
		}
		// The arg to epoll_create is a hint to the kernel
		// about the number of FDs we will care about.
		// We don't know, and since 2.6.8 the kernel ignores it anyhow.
		if p.epfd, err = syscall.EpollCreate(16); err != nil {
			return nil, os.NewSyscallError("epoll_create", err)
		}
		syscall.CloseOnExec(p.epfd)
	}
	p.events = make(map[int]uint32)
	return p, nil
}

func (p *pollster) AddFD(fd int, mode int, repeat bool) (bool, error) {
	// pollServer is locked.

	var already bool
	p.ctlEvent.Fd = int32(fd)
	p.ctlEvent.Events, already = p.events[fd]
	if !repeat {
		p.ctlEvent.Events |= syscall.EPOLLONESHOT
	}
	if mode == 'r' {
		p.ctlEvent.Events |= readFlags
	} else {
		p.ctlEvent.Events |= writeFlags
	}

	var op int
	if already {
		op = syscall.EPOLL_CTL_MOD
	} else {
		op = syscall.EPOLL_CTL_ADD
	}
	if err := syscall.EpollCtl(p.epfd, op, fd, &p.ctlEvent); err != nil {
		return false, os.NewSyscallError("epoll_ctl", err)
	}
	p.events[fd] = p.ctlEvent.Events
	return false, nil
}

func (p *pollster) StopWaiting(fd int, bits uint) {
	// pollServer is locked.

	events, already := p.events[fd]
	if !already {
		// The fd returned by the kernel may have been
		// cancelled already; return silently.
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
		p.ctlEvent.Fd = int32(fd)
		p.ctlEvent.Events = events
		if err := syscall.EpollCtl(p.epfd, syscall.EPOLL_CTL_MOD, fd, &p.ctlEvent); err != nil {
			print("Epoll modify fd=", fd, ": ", err.Error(), "\n")
		}
		p.events[fd] = events
	} else {
		if err := syscall.EpollCtl(p.epfd, syscall.EPOLL_CTL_DEL, fd, nil); err != nil {
			print("Epoll delete fd=", fd, ": ", err.Error(), "\n")
		}
		delete(p.events, fd)
	}
}

func (p *pollster) DelFD(fd int, mode int) {
	// pollServer is locked.

	if mode == 'r' {
		p.StopWaiting(fd, readFlags)
	} else {
		p.StopWaiting(fd, writeFlags)
	}

	// Discard any queued up events.
	i := 0
	for i < len(p.waitEvents) {
		if fd == int(p.waitEvents[i].Fd) {
			copy(p.waitEvents[i:], p.waitEvents[i+1:])
			p.waitEvents = p.waitEvents[:len(p.waitEvents)-1]
		} else {
			i++
		}
	}
}

func (p *pollster) WaitFD(s *pollServer, nsec int64) (fd int, mode int, err error) {
	for len(p.waitEvents) == 0 {
		var msec int = -1
		if nsec > 0 {
			msec = int((nsec + 1e6 - 1) / 1e6)
		}

		s.Unlock()
		n, err := syscall.EpollWait(p.epfd, p.waitEventBuf[0:], msec)
		s.Lock()

		if err != nil {
			if err == syscall.EAGAIN || err == syscall.EINTR {
				continue
			}
			return -1, 0, os.NewSyscallError("epoll_wait", err)
		}
		if n == 0 {
			return -1, 0, nil
		}
		p.waitEvents = p.waitEventBuf[0:n]
	}

	ev := &p.waitEvents[0]
	p.waitEvents = p.waitEvents[1:]

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

func (p *pollster) Close() error {
	return os.NewSyscallError("close", syscall.Close(p.epfd))
}

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Waiting for FDs via epoll(7).

package net

import (
	"net";
	"os";
	"syscall";
)

const (
	Read = syscall.EPOLLIN | syscall.EPOLLRDHUP;
	Write = syscall.EPOLLOUT
)

export type Pollster struct {
	epfd int64;

	// Events we're already waiting for
	events map[int64] uint32;
}

export func NewPollster() (p *Pollster, err *os.Error) {
	p = new(*Pollster);
	var e int64;

	// The arg to epoll_create is a hint to the kernel
	// about the number of FDs we will care about.
	// We don't know.
	if p.epfd, e = syscall.epoll_create(16); e != 0 {
		return nil, os.ErrnoToError(e)
	}
	p.events = new(map[int64] uint32);
	return p, nil
}

func (p *Pollster) AddFD(fd int64, mode int, repeat bool) *os.Error {
	var ev syscall.EpollEvent;
	var already bool;
	ev.fd = int32(fd);
	ev.events, already = p.events[fd];
	if !repeat {
		ev.events |= syscall.EPOLLONESHOT
	}
	if mode == 'r' {
		ev.events |= Read
	} else {
		ev.events |= Write
	}

	var op int64;
	if already {
		op = syscall.EPOLL_CTL_MOD
	} else {
		op = syscall.EPOLL_CTL_ADD
	}
	if e := syscall.epoll_ctl(p.epfd, op, fd, &ev); e != 0 {
		return os.ErrnoToError(e)
	}
	p.events[fd] = ev.events;
	return nil
}

func (p *Pollster) StopWaiting(fd int64, bits uint) {
	events, already := p.events[fd];
	if !already {
		print("Epoll unexpected fd=", fd, "\n");
		return
	}

	// If syscall.EPOLLONESHOT is not set, the wait
	// is a repeating wait, so don't change it.
	if events & syscall.EPOLLONESHOT == 0 {
		return
	}

	// Disable the given bits.
	// If we're still waiting for other events, modify the fd
	// event in the kernel.  Otherwise, delete it.
	events &= ^uint32(bits);
	if int32(events) & ^syscall.EPOLLONESHOT != 0 {
		var ev syscall.EpollEvent;
		ev.fd = int32(fd);
		ev.events = events;
		if e := syscall.epoll_ctl(p.epfd, syscall.EPOLL_CTL_MOD, fd, &ev); e != 0 {
			print("Epoll modify fd=", fd, ": ", os.ErrnoToError(e).String(), "\n")
		}
		p.events[fd] = events
	} else {
		if e := syscall.epoll_ctl(p.epfd, syscall.EPOLL_CTL_DEL, fd, nil); e != 0 {
			print("Epoll delete fd=", fd, ": ", os.ErrnoToError(e).String(), "\n")
		}
		p.events[fd] = 0, false
	}
}

func (p *Pollster) WaitFD() (fd int64, mode int, err *os.Error) {
	// Get an event.
	var evarray [1]syscall.EpollEvent;
	ev := &evarray[0];
	n, e := syscall.epoll_wait(p.epfd, evarray, -1);
	for e == syscall.EAGAIN || e == syscall.EINTR {
		n, e = syscall.epoll_wait(p.epfd, evarray, -1)
	}
	if e != 0 {
		return -1, 0, os.ErrnoToError(e)
	}
	fd = int64(ev.fd);

	if ev.events & Write != 0 {
		p.StopWaiting(fd, Write);
		return fd, 'w', nil
	}
	if ev.events & Read != 0 {
		p.StopWaiting(fd, Read);
		return fd, 'r', nil
	}

	// Other events are error conditions - wake whoever is waiting.
	events, already := p.events[fd];
	if events & Write != 0 {
		p.StopWaiting(fd, Write);
		return fd, 'w', nil
	}
	p.StopWaiting(fd, Read);
	return fd, 'r', nil
}

func (p *Pollster) Close() *os.Error {
	r, e := syscall.close(p.epfd);
	return os.ErrnoToError(e)
}

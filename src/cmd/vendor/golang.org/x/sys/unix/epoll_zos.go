// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build zos && s390x

package unix

import (
	"sync"
)

// This file simulates epoll on z/OS using poll.

// Analogous to epoll_event on Linux.
// TODO(neeilan): Pad is because the Linux kernel expects a 96-bit struct. We never pass this to the kernel; remove?
type EpollEvent struct {
	Events uint32
	Fd     int32
	Pad    int32
}

const (
	EPOLLERR      = 0x8
	EPOLLHUP      = 0x10
	EPOLLIN       = 0x1
	EPOLLMSG      = 0x400
	EPOLLOUT      = 0x4
	EPOLLPRI      = 0x2
	EPOLLRDBAND   = 0x80
	EPOLLRDNORM   = 0x40
	EPOLLWRBAND   = 0x200
	EPOLLWRNORM   = 0x100
	EPOLL_CTL_ADD = 0x1
	EPOLL_CTL_DEL = 0x2
	EPOLL_CTL_MOD = 0x3
	// The following constants are part of the epoll API, but represent
	// currently unsupported functionality on z/OS.
	// EPOLL_CLOEXEC  = 0x80000
	// EPOLLET        = 0x80000000
	// EPOLLONESHOT   = 0x40000000
	// EPOLLRDHUP     = 0x2000     // Typically used with edge-triggered notis
	// EPOLLEXCLUSIVE = 0x10000000 // Exclusive wake-up mode
	// EPOLLWAKEUP    = 0x20000000 // Relies on Linux's BLOCK_SUSPEND capability
)

// TODO(neeilan): We can eliminate these epToPoll / pToEpoll calls by using identical mask values for POLL/EPOLL
// constants where possible The lower 16 bits of epoll events (uint32) can fit any system poll event (int16).

// epToPollEvt converts epoll event field to poll equivalent.
// In epoll, Events is a 32-bit field, while poll uses 16 bits.
func epToPollEvt(events uint32) int16 {
	var ep2p = map[uint32]int16{
		EPOLLIN:  POLLIN,
		EPOLLOUT: POLLOUT,
		EPOLLHUP: POLLHUP,
		EPOLLPRI: POLLPRI,
		EPOLLERR: POLLERR,
	}

	var pollEvts int16 = 0
	for epEvt, pEvt := range ep2p {
		if (events & epEvt) != 0 {
			pollEvts |= pEvt
		}
	}

	return pollEvts
}

// pToEpollEvt converts 16 bit poll event bitfields to 32-bit epoll event fields.
func pToEpollEvt(revents int16) uint32 {
	var p2ep = map[int16]uint32{
		POLLIN:  EPOLLIN,
		POLLOUT: EPOLLOUT,
		POLLHUP: EPOLLHUP,
		POLLPRI: EPOLLPRI,
		POLLERR: EPOLLERR,
	}

	var epollEvts uint32 = 0
	for pEvt, epEvt := range p2ep {
		if (revents & pEvt) != 0 {
			epollEvts |= epEvt
		}
	}

	return epollEvts
}

// Per-process epoll implementation.
type epollImpl struct {
	mu       sync.Mutex
	epfd2ep  map[int]*eventPoll
	nextEpfd int
}

// eventPoll holds a set of file descriptors being watched by the process. A process can have multiple epoll instances.
// On Linux, this is an in-kernel data structure accessed through a fd.
type eventPoll struct {
	mu  sync.Mutex
	fds map[int]*EpollEvent
}

// epoll impl for this process.
var impl epollImpl = epollImpl{
	epfd2ep:  make(map[int]*eventPoll),
	nextEpfd: 0,
}

func (e *epollImpl) epollcreate(size int) (epfd int, err error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	epfd = e.nextEpfd
	e.nextEpfd++

	e.epfd2ep[epfd] = &eventPoll{
		fds: make(map[int]*EpollEvent),
	}
	return epfd, nil
}

func (e *epollImpl) epollcreate1(flag int) (fd int, err error) {
	return e.epollcreate(4)
}

func (e *epollImpl) epollctl(epfd int, op int, fd int, event *EpollEvent) (err error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	ep, ok := e.epfd2ep[epfd]
	if !ok {

		return EBADF
	}

	switch op {
	case EPOLL_CTL_ADD:
		// TODO(neeilan): When we make epfds and fds disjoint, detect epoll
		// loops here (instances watching each other) and return ELOOP.
		if _, ok := ep.fds[fd]; ok {
			return EEXIST
		}
		ep.fds[fd] = event
	case EPOLL_CTL_MOD:
		if _, ok := ep.fds[fd]; !ok {
			return ENOENT
		}
		ep.fds[fd] = event
	case EPOLL_CTL_DEL:
		if _, ok := ep.fds[fd]; !ok {
			return ENOENT
		}
		delete(ep.fds, fd)

	}
	return nil
}

// Must be called while holding ep.mu
func (ep *eventPoll) getFds() []int {
	fds := make([]int, len(ep.fds))
	for fd := range ep.fds {
		fds = append(fds, fd)
	}
	return fds
}

func (e *epollImpl) epollwait(epfd int, events []EpollEvent, msec int) (n int, err error) {
	e.mu.Lock() // in [rare] case of concurrent epollcreate + epollwait
	ep, ok := e.epfd2ep[epfd]

	if !ok {
		e.mu.Unlock()
		return 0, EBADF
	}

	pollfds := make([]PollFd, 4)
	for fd, epollevt := range ep.fds {
		pollfds = append(pollfds, PollFd{Fd: int32(fd), Events: epToPollEvt(epollevt.Events)})
	}
	e.mu.Unlock()

	n, err = Poll(pollfds, msec)
	if err != nil {
		return n, err
	}

	i := 0
	for _, pFd := range pollfds {
		if pFd.Revents != 0 {
			events[i] = EpollEvent{Fd: pFd.Fd, Events: pToEpollEvt(pFd.Revents)}
			i++
		}

		if i == n {
			break
		}
	}

	return n, nil
}

func EpollCreate(size int) (fd int, err error) {
	return impl.epollcreate(size)
}

func EpollCreate1(flag int) (fd int, err error) {
	return impl.epollcreate1(flag)
}

func EpollCtl(epfd int, op int, fd int, event *EpollEvent) (err error) {
	return impl.epollctl(epfd, op, fd, event)
}

// Because EpollWait mutates events, the caller is expected to coordinate
// concurrent access if calling with the same epfd from multiple goroutines.
func EpollWait(epfd int, events []EpollEvent, msec int) (n int, err error) {
	return impl.epollwait(epfd, events, msec)
}

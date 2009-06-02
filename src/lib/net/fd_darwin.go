// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Waiting for FDs via kqueue/kevent.

package net

import (
	"net";
	"os";
	"syscall";
)

var kqueuePhaseError = &Error{"kqueue phase error"}

type pollster struct {
	kq int;
	eventbuf [10]syscall.Kevent_t;
	events []syscall.Kevent_t;
}

func newpollster() (p *pollster, err os.Error) {
	p = new(pollster);
	var e int;
	if p.kq, e = syscall.Kqueue(); e != 0 {
		return nil, os.ErrnoToError(e)
	}
	p.events = p.eventbuf[0:0];
	return p, nil
}

func (p *pollster) AddFD(fd int, mode int, repeat bool) os.Error {
	var kmode int16;
	if mode == 'r' {
		kmode = syscall.EVFILT_READ
	} else {
		kmode = syscall.EVFILT_WRITE
	}
	var events [1]syscall.Kevent_t;
	ev := &events[0];
	ev.Ident = uint64(fd);
	ev.Filter = kmode;

	// EV_ADD - add event to kqueue list
	// EV_RECEIPT - generate fake EV_ERROR as result of add,
	//	rather than waiting for real event
	// EV_ONESHOT - delete the event the first time it triggers
	ev.Flags = syscall.EV_ADD | syscall.EV_RECEIPT;
	if !repeat {
		ev.Flags |= syscall.EV_ONESHOT
	}

	n, e := syscall.Kevent(p.kq, &events, &events, nil);
	if e != 0 {
		return os.ErrnoToError(e)
	}
	if n != 1 || (ev.Flags & syscall.EV_ERROR) == 0 || ev.Ident != uint64(fd) || ev.Filter != kmode {
		return kqueuePhaseError
	}
	if ev.Data != 0 {
		return os.ErrnoToError(int(ev.Data))
	}
	return nil
}

func (p *pollster) DelFD(fd int, mode int) {
	var kmode int16;
	if mode == 'r' {
		kmode = syscall.EVFILT_READ
	} else {
		kmode = syscall.EVFILT_WRITE
	}
	var events [1]syscall.Kevent_t;
	ev := &events[0];
	ev.Ident = uint64(fd);
	ev.Filter = kmode;

	// EV_DELETE - delete event from kqueue list
	// EV_RECEIPT - generate fake EV_ERROR as result of add,
	//	rather than waiting for real event
	ev.Flags = syscall.EV_DELETE | syscall.EV_RECEIPT;
	syscall.Kevent(p.kq, &events, &events, nil);
}

func (p *pollster) WaitFD(nsec int64) (fd int, mode int, err os.Error) {
	var t *syscall.Timespec;
	for len(p.events) == 0 {
		if nsec > 0 {
			if t == nil {
				t = new(syscall.Timespec);
			}
			t.Sec = nsec / 1e9;
			t.Nsec = int64(nsec % 1e9);
//			*t = syscall.NsecToTimespec(nsec);
		}
		nn, e := syscall.Kevent(p.kq, nil, &p.eventbuf, t);
		if e != 0 {
			if e == syscall.EINTR {
				continue
			}
			return -1, 0, os.ErrnoToError(e)
		}
		if nn == 0 {
			return -1, 0, nil;
		}
		p.events = p.eventbuf[0:nn]
	}
	ev := &p.events[0];
	p.events = p.events[1:len(p.events)];
	fd = int(ev.Ident);
	if ev.Filter == syscall.EVFILT_READ {
		mode = 'r'
	} else {
		mode = 'w'
	}
	return fd, mode, nil
}

func (p *pollster) Close() os.Error {
	return os.ErrnoToError(syscall.Close(p.kq))
}

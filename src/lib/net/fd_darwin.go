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

export type Pollster struct {
	kq int64;
	eventbuf [10]syscall.Kevent;
	events []syscall.Kevent;
}

export func NewPollster() (p *Pollster, err *os.Error) {
	p = new(*Pollster);
	var e int64;
	if p.kq, e = syscall.kqueue(); e != 0 {
		return nil, os.ErrnoToError(e)
	}
	p.events = p.eventbuf[0:0];
	return p, nil
}

func (p *Pollster) AddFD(fd int64, mode int, repeat bool) *os.Error {
	var kmode int16;
	if mode == 'r' {
		kmode = syscall.EVFILT_READ
	} else {
		kmode = syscall.EVFILT_WRITE
	}
	var events [1]syscall.Kevent;
	ev := &events[0];
	ev.ident = fd;
	ev.filter = kmode;

	// EV_ADD - add event to kqueue list
	// EV_RECEIPT - generate fake EV_ERROR as result of add,
	//	rather than waiting for real event
	// EV_ONESHOT - delete the event the first time it triggers
	ev.flags = syscall.EV_ADD | syscall.EV_RECEIPT;
	if !repeat {
		ev.flags |= syscall.EV_ONESHOT
	}

	n, e := syscall.kevent(p.kq, events, events, nil);
	if e != 0 {
		return os.ErrnoToError(e)
	}
	if n != 1 || (ev.flags & syscall.EV_ERROR) == 0 || ev.ident != fd || ev.filter != kmode {
		return os.NewError("kqueue phase error")
	}
	if ev.data != 0 {
		return os.ErrnoToError(ev.data)
	}
	return nil
}

func (p *Pollster) WaitFD() (fd int64, mode int, err *os.Error) {
	for len(p.events) == 0 {
		nn, e := syscall.kevent(p.kq, nil, p.eventbuf, nil);
		if e != 0 {
			if e == syscall.EAGAIN || e == syscall.EINTR {
				continue
			}
			return -1, 0, os.ErrnoToError(e)
		}
		p.events = p.eventbuf[0:nn]
	}
	ev := &p.events[0];
	p.events = p.events[1:len(p.events)];
	fd = ev.ident;
	if ev.filter == syscall.EVFILT_READ {
		mode = 'r'
	} else {
		mode = 'w'
	}
	return fd, mode, nil
}

func (p *Pollster) Close() *os.Error {
	r, e := syscall.close(p.kq);
	return os.ErrnoToError(e)
}

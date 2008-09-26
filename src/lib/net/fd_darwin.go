// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(rsc): All the prints in this file should go to standard error.

package net

import (
	"net";
	"once";
	"os";
	"syscall";
)

const Debug = false

// Network file descriptor.  Only intended to be used internally,
// but have to export to make it available in other files implementing package net.
export type FD struct {
	fd int64;
	cr *chan *FD;
	cw *chan *FD;

	// owned by fd wait server
	ncr, ncw int;
	next *FD;
}

func WaitRead(fd *FD);
func WaitWrite(fd *FD);
func StartServer();

func MakeNonblocking(fd int64) *os.Error {
	if Debug { print("MakeNonBlocking ", fd, "\n") }
	flags, e := syscall.fcntl(fd, syscall.F_GETFL, 0)
	if e != 0 {
		return os.ErrnoToError(e)
	}
	flags, e = syscall.fcntl(fd, syscall.F_SETFL, flags | syscall.O_NONBLOCK)
	if e != 0 {
		return os.ErrnoToError(e)
	}
	return nil
}

export func NewFD(fd int64) (f *FD, err *os.Error) {
	once.Do(&StartServer);
	if err = MakeNonblocking(fd); err != nil {
		return nil, err
	}
	f = new(FD);
	f.fd = fd;
	f.cr = new(chan *FD);
	f.cw = new(chan *FD);
	return f, nil
}

func (fd *FD) Close() *os.Error {
	if fd == nil {
		return os.EINVAL
	}
	r1, e := syscall.close(fd.fd);
	if e != 0 {
		return os.ErrnoToError(e)
	}
	return nil
}

func (fd *FD) Read(p *[]byte) (n int, err *os.Error) {
	if fd == nil {
		return -1, os.EINVAL
	}
L:	nn, e := syscall.read(fd.fd, &p[0], int64(len(p)))
	switch {
	case e == syscall.EAGAIN:
		WaitRead(fd)
		goto L
	case e != 0:
		return -1, os.ErrnoToError(e)
	}
	return int(nn), nil
}

func (fd *FD) Write(p *[]byte) (n int, err *os.Error) {
	if fd == nil {
		return -1, os.EINVAL
	}
	total := len(p)
	for len(p) > 0 {
	L:	nn, e := syscall.write(fd.fd, &p[0], int64(len(p)))
		switch {
		case e == syscall.EAGAIN:
			WaitWrite(fd)
			goto L
		case e != 0:
			return total - len(p), os.ErrnoToError(e)
		}
		p = p[nn:len(p)]
	}
	return total, nil
}

func (fd *FD) Accept(sa *syscall.Sockaddr) (nfd *FD, err *os.Error) {
	if fd == nil {
		return nil, os.EINVAL
	}
L:	s, e := syscall.accept(fd.fd, sa)
	switch {
	case e == syscall.EAGAIN:
		WaitRead(fd)
		goto L
	case e != 0:
		return nil, os.ErrnoToError(e)
	}
	nfd, err = NewFD(s)
	if err != nil {
		syscall.close(s)
		return nil, err
	}
	return nfd, nil
}


// Waiting for FDs via kqueue(2).
type Kstate struct {
	cr *chan *FD;
	cw *chan *FD;
	pr *os.FD;
	pw *os.FD;
	pend *map[int64] *FD;
	kq int64;
}

var kstate Kstate;

func KqueueAdd(fd int64, mode byte, repeat bool) *os.Error {
	if Debug { print("Kqueue add ", fd, " ", mode, " ", repeat, "\n") }
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
	// EV_RECEIPT - generate fake EV_ERROR as result of add
	// EV_ONESHOT - delete the event the first time it triggers
	ev.flags = syscall.EV_ADD | syscall.EV_RECEIPT
	if !repeat {
		ev.flags |= syscall.EV_ONESHOT
	}

	n, e := syscall.kevent(kstate.kq, &events, &events, nil);
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

func KqueueAddFD(fd *FD, mode byte) *os.Error {
	if e := KqueueAdd(fd.fd, 'r', false); e != nil {
		return e
	}
	id := fd.fd << 1
	if mode == 'r' {
		fd.ncr++
	} else {
		id++
		fd.ncw++
	}
	kstate.pend[id] = fd
	return nil
}

func KqueueGet(events *[]syscall.Kevent) (n int, err *os.Error) {
	var nn, e int64;
	if nn, e = syscall.kevent(kstate.kq, nil, events, nil); e != 0 {
		return -1, os.ErrnoToError(e)
	}
	return int(nn),  nil
}

func KqueueLookup(ev *syscall.Kevent) (fd *FD, mode byte) {
	id := ev.ident << 1
	if ev.filter == syscall.EVFILT_READ {
		mode = 'r'
	} else {
		id++
		mode = 'w'
	}
	var ok bool
	if fd, ok = kstate.pend[id]; !ok {
		return nil, 0
	}
	kstate.pend[id] = nil, false
	return fd, mode
}

func Serve() {
	var r, e int64;
	k := &kstate;

	if Debug { print("Kqueue server running\n") }
	var events [10]syscall.Kevent;
	var scratch [100]byte;
	for {
		var n int
		var err *os.Error;
		if n, err = KqueueGet(&events); err != nil {
			print("kqueue get: ", err.String(), "\n")
			return
		}
		if Debug { print("Kqueue server get ", n, "\n") }
		for i := 0; i < n; i++ {
			ev := &events[i]
			if ev.ident == k.pr.fd {
				if Debug { print("Kqueue server wakeup\n") }
				// Drain our wakeup pipe
				for {
					nn, e := k.pr.Read(&scratch)
					if Debug { print("Read ", k.pr.fd, " ", nn, " ", e.String(), "\n") }
					if nn <= 0 {
						break
					}
				}

				if Debug { print("Kqueue server drain channels\n") }
				// Then read from channels.
				for {
					fd, ok := <-k.cr
					if !ok {
						break
					}
					KqueueAddFD(fd, 'r')
				}
				for {
					fd, ok := <-k.cw
					if !ok {
						break
					}
					KqueueAddFD(fd, 'w')
				}
				if Debug { print("Kqueue server awake\n") }
				continue
			}

			// Otherwise, wakeup the right FD.
			fd, mode := KqueueLookup(ev);
			if fd == nil {
				print("kqueue: unexpected wakeup for fd=", ev.ident, " filter=", ev.filter, "\n")
				continue
			}
			if mode == 'r' {
				if Debug { print("Kqueue server r fd=", fd.fd, " ncr=", fd.ncr, "\n") }
				for fd.ncr > 0 {
					fd.ncr--
					fd.cr <- fd
				}
			} else {
				if Debug { print("Kqueue server w fd=", fd.fd, " ncw=", fd.ncw, "\n") }
				for fd.ncw > 0 {
					fd.ncw--
					fd.cw <- fd
				}
			}
		}
	}
}

func StartServer() {
	k := &kstate;

	k.cr = new(chan *FD, 1);
	k.cw = new(chan *FD, 1);
	k.pend = new(map[int64] *FD)

	var err *os.Error
	if k.pr, k.pw, err = os.Pipe(); err != nil {
		print("kqueue pipe: ", err.String(), "\n")
		return
	}

	if err := MakeNonblocking(k.pr.fd); err != nil {
		print("make nonblocking pr: ", err.String(), "\n")
		return
	}
	if err := MakeNonblocking(k.pw.fd); err != nil {
		print("make nonblocking pw: ", err.String(), "\n")
		return
	}

	var e int64
	if k.kq, e = syscall.kqueue(); e != 0 {
		err := os.ErrnoToError(e);
		print("kqueue: ", err.String(), "\n")
		return
	}

	if err := KqueueAdd(k.pr.fd, 'r', true); err != nil {
		print("kqueue add pipe: ", err.String(), "\n")
		return
	}

	go Serve()
}

func WakeupServer() {
	var b [1]byte;
	kstate.pw.Write(&b);
}

func WaitRead(fd *FD) {
	kstate.cr <- fd;
	WakeupServer();
	<-fd.cr
}

func WaitWrite(fd *FD) {
	kstate.cw <- fd;
	WakeupServer();
	<-fd.cw
}

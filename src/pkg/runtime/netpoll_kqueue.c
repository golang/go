// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"

// Integrated network poller (kqueue-based implementation).

int32	runtime·kqueue(void);
int32	runtime·kevent(int32, Kevent*, int32, Kevent*, int32, Timespec*);
void	runtime·closeonexec(int32);

static int32 kq = -1;

void
runtime·netpollinit(void)
{
	kq = runtime·kqueue();
	if(kq < 0) {
		runtime·printf("netpollinit: kqueue failed with %d\n", -kq);
		runtime·throw("netpollinit: kqueue failed");
	}
	runtime·closeonexec(kq);
}

int32
runtime·netpollopen(int32 fd, PollDesc *pd)
{
	Kevent ev[2];
	int32 n;

	// Arm both EVFILT_READ and EVFILT_WRITE in edge-triggered mode (EV_CLEAR)
	// for the whole fd lifetime.  The notifications are automatically unregistered
	// when fd is closed.
	ev[0].ident = fd;
	ev[0].filter = EVFILT_READ;
	ev[0].flags = EV_ADD|EV_RECEIPT|EV_CLEAR;
	ev[0].fflags = 0;
	ev[0].data = 0;
	ev[0].udata = (byte*)pd;
	ev[1] = ev[0];
	ev[1].filter = EVFILT_WRITE;
	n = runtime·kevent(kq, ev, 2, ev, 2, nil);
	if(n < 0)
		return -n;
	if(n != 2 ||
		(ev[0].flags&EV_ERROR) == 0 || ev[0].ident != fd || ev[0].filter != EVFILT_READ ||
		(ev[1].flags&EV_ERROR) == 0 || ev[1].ident != fd || ev[1].filter != EVFILT_WRITE)
		return EFAULT;  // just to mark out from other errors
	if(ev[0].data != 0)
		return ev[0].data;
	if(ev[1].data != 0)
		return ev[1].data;
	return 0;
}

int32
runtime·netpollclose(int32 fd)
{
	// Don't need to unregister because calling close()
	// on fd will remove any kevents that reference the descriptor.
	USED(fd);
	return 0;
}

// Polls for ready network connections.
// Returns list of goroutines that become runnable.
G*
runtime·netpoll(bool block)
{
	static int32 lasterr;
	Kevent events[64], *ev;
	Timespec ts, *tp;
	int32 n, i;
	G *gp;

	if(kq == -1)
		return nil;
	tp = nil;
	if(!block) {
		ts.tv_sec = 0;
		ts.tv_nsec = 0;
		tp = &ts;
	}
	gp = nil;
retry:
	n = runtime·kevent(kq, nil, 0, events, nelem(events), tp);
	if(n < 0) {
		if(n != -EINTR && n != lasterr) {
			lasterr = n;
			runtime·printf("runtime: kevent on fd %d failed with %d\n", kq, -n);
		}
		goto retry;
	}
	for(i = 0; i < n; i++) {
		ev = &events[i];
		if(ev->filter == EVFILT_READ)
			runtime·netpollready(&gp, (PollDesc*)ev->udata, 'r');
		if(ev->filter == EVFILT_WRITE)
			runtime·netpollready(&gp, (PollDesc*)ev->udata, 'w');
	}
	if(block && gp == nil)
		goto retry;
	return gp;
}

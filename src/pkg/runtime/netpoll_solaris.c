// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"

#pragma dynimport libc·fcntl fcntl "libc.so"
#pragma dynimport libc·port_create port_create "libc.so"
#pragma dynimport libc·port_associate port_associate "libc.so"
#pragma dynimport libc·port_dissociate port_dissociate "libc.so"
#pragma dynimport libc·port_getn port_getn "libc.so"
extern uintptr libc·fcntl;
extern uintptr libc·port_create;
extern uintptr libc·port_associate;
extern uintptr libc·port_dissociate;
extern uintptr libc·port_getn;

#define errno (*m->perrno)

int32
runtime·fcntl(int32 fd, int32 cmd, uintptr arg)
{
	return runtime·sysvicall6(libc·fcntl, 3,
	    (uintptr)fd, (uintptr)cmd, (uintptr)arg);
}

int32
runtime·port_create(void)
{
	return runtime·sysvicall6(libc·port_create, 0);
}

int32
runtime·port_associate(int32 port, int32 source, uintptr object, int32 events, uintptr user)
{
	return runtime·sysvicall6(libc·port_associate,
	    5, (uintptr)port, (uintptr)source, object, (uintptr)events, user);
}

int32
runtime·port_dissociate(int32 port, int32 source, uintptr object)
{
	return runtime·sysvicall6(libc·port_dissociate,
	    3, (uintptr)port, (uintptr)source, object);
}

int32
runtime·port_getn(int32 port, PortEvent *evs, uint32 max, uint32 *nget, Timespec *timeout)
{
	return runtime·sysvicall6(libc·port_getn, 5, (uintptr)port,
	    (uintptr)evs, (uintptr)max, (uintptr)nget, (uintptr)timeout);
}

static int32 portfd = -1;

void
runtime·netpollinit(void)
{
	if((portfd = runtime·port_create()) >= 0) {
		runtime·fcntl(portfd, F_SETFD, FD_CLOEXEC);
		return;
	}

	runtime·printf("netpollinit: failed to create port (%d)\n", errno);
	runtime·throw("netpollinit: failed to create port");
}

int32
runtime·netpollopen(uintptr fd, PollDesc *pd)
{
	uint32 events = POLLIN | POLLOUT;
	*runtime·netpolluser(pd) = (void*)events;

	return runtime·port_associate(portfd, PORT_SOURCE_FD, fd, events, (uintptr)pd);
}

int32
runtime·netpollclose(uintptr fd)
{
	return runtime·port_dissociate(portfd, PORT_SOURCE_FD, fd);
}

void
runtime·netpollupdate(PollDesc* pd, uint32 set, uint32 clear)
{
	uint32 *ep, old, events;
	uintptr fd = runtime·netpollfd(pd);
	ep = (uint32*)runtime·netpolluser(pd);

	do {
		old = *ep;
		events = (old & ~clear) | set;
		if(old == events)
			return;

		if(events && runtime·port_associate(portfd, PORT_SOURCE_FD, fd, events, (uintptr)pd) != 0) {
			runtime·printf("netpollupdate: failed to associate (%d)\n", errno);
			runtime·throw("netpollupdate: failed to associate");
		}
	} while(runtime·cas(ep, old, events) != events);
}

void
runtime·netpollarm(PollDesc* pd, int32 mode)
{
	switch(mode) {
	case 'r':
		runtime·netpollupdate(pd, POLLIN, 0);
		break;
	case 'w':
		runtime·netpollupdate(pd, POLLOUT, 0);
		break;
	default:
		runtime·throw("netpollarm: bad mode");
	}
}

// polls for ready network connections
// returns list of goroutines that become runnable
G*
runtime·netpoll(bool block)
{
	static int32 lasterr;
	PortEvent events[128], *ev;
	PollDesc *pd;
	int32 i, mode;
	uint32 n;
	Timespec *wait = nil, zero;
	G *gp;

	if(portfd == -1)
		return (nil);

	if(!block) {
		zero.tv_sec = 0;
		zero.tv_nsec = 0;
		wait = &zero;
	}

retry:
	n = 1;

	if(runtime·port_getn(portfd, events, nelem(events), &n, wait) < 0) {
		if(errno != EINTR && errno != lasterr) {
			lasterr = errno;
			runtime·printf("runtime: port_getn on fd %d "
			    "failed with %d\n", portfd, errno);
		}
		goto retry;
	}

	gp = nil;

	for(i = 0; i < n; i++) {
		ev = &events[i];

		if(ev->portev_events == 0)
			continue;

		if((pd = (PollDesc *)ev->portev_user) == nil)
			continue;

		mode = 0;

		if(ev->portev_events & (POLLIN|POLLHUP|POLLERR))
			mode += 'r';

		if(ev->portev_events & (POLLOUT|POLLHUP|POLLERR))
			mode += 'w';

		//
		// To effect edge-triggered events, we need to be sure to
		// update our association with whatever events were not
		// set with the event.
		//
		runtime·netpollupdate(pd, 0, ev->portev_events & (POLLIN|POLLOUT));

		if(mode)
			runtime·netpollready(&gp, pd, mode);
	}

	if(block && gp == nil)
		goto retry;
	return gp;
}

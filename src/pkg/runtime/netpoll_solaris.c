// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"

// Solaris runtime-integrated network poller.
// 
// Solaris uses event ports for scalable network I/O. Event
// ports are level-triggered, unlike epoll and kqueue which
// can be configured in both level-triggered and edge-triggered
// mode. Level triggering means we have to keep track of a few things
// ourselves. After we receive an event for a file descriptor,
// it's our responsibility to ask again to be notified for future
// events for that descriptor. When doing this we must keep track of
// what kind of events the goroutines are currently interested in,
// for example a fd may be open both for reading and writing.
// 
// A description of the high level operation of this code
// follows. Networking code will get a file descriptor by some means
// and will register it with the netpolling mechanism by a code path
// that eventually calls runtime·netpollopen. runtime·netpollopen
// calls port_associate with an empty event set. That means that we
// will not receive any events at this point. The association needs
// to be done at this early point because we need to process the I/O
// readiness notification at some point in the future. If I/O becomes
// ready when nobody is listening, when we finally care about it,
// nobody will tell us anymore.
// 
// Beside calling runtime·netpollopen, the networking code paths
// will call runtime·netpollarm each time goroutines are interested
// in doing network I/O. Because now we know what kind of I/O we
// are interested in (reading/writting), we can call port_associate
// passing the correct type of event set (POLLIN/POLLOUT). As we made
// sure to have already associated the file descriptor with the port,
// when we now call port_associate, we will unblock the main poller
// loop (in runtime·netpoll) right away if the socket is actually
// ready for I/O.
// 
// The main poller loop runs in its own thread waiting for events
// using port_getn. When an event happens, it will tell the scheduler
// about it using runtime·netpollready. Besides doing this, it must
// also re-associate the events that were not part of this current
// notification with the file descriptor. Failing to do this would
// mean each notification will prevent concurrent code using the
// same file descriptor in parallel.
// 
// The logic dealing with re-associations is encapsulated in
// runtime·netpollupdate. This function takes care to associate the
// descriptor only with the subset of events that were previously
// part of the association, except the one that just happened. We
// can't re-associate with that right away, because event ports
// are level triggered so it would cause a busy loop. Instead, that
// association is effected only by the runtime·netpollarm code path,
// when Go code actually asks for I/O.
// 
// The open and arming mechanisms are serialized using the lock
// inside PollDesc. This is required because the netpoll loop runs
// asynchonously in respect to other Go code and by the time we get
// to call port_associate to update the association in the loop, the
// file descriptor might have been closed and reopened already. The
// lock allows runtime·netpollupdate to be called synchronously from
// the loop thread while preventing other threads operating to the
// same PollDesc, so once we unblock in the main loop, until we loop
// again we know for sure we are always talking about the same file
// descriptor and can safely access the data we want (the event set).

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

#define errno (*g->m->perrno)

int32
runtime·fcntl(int32 fd, int32 cmd, uintptr arg)
{
	return runtime·sysvicall3(libc·fcntl, (uintptr)fd, (uintptr)cmd, (uintptr)arg);
}

int32
runtime·port_create(void)
{
	return runtime·sysvicall0(libc·port_create);
}

int32
runtime·port_associate(int32 port, int32 source, uintptr object, int32 events, uintptr user)
{
	return runtime·sysvicall5(libc·port_associate, (uintptr)port, (uintptr)source, object, (uintptr)events, user);
}

int32
runtime·port_dissociate(int32 port, int32 source, uintptr object)
{
	return runtime·sysvicall3(libc·port_dissociate, (uintptr)port, (uintptr)source, object);
}

int32
runtime·port_getn(int32 port, PortEvent *evs, uint32 max, uint32 *nget, Timespec *timeout)
{
	return runtime·sysvicall5(libc·port_getn, (uintptr)port, (uintptr)evs, (uintptr)max, (uintptr)nget, (uintptr)timeout);
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
	int32 r;

	runtime·netpolllock(pd);
	// We don't register for any specific type of events yet, that's
	// netpollarm's job. We merely ensure we call port_associate before
	// asynchonous connect/accept completes, so when we actually want
	// to do any I/O, the call to port_associate (from netpollarm,
	// with the interested event set) will unblock port_getn right away
	// because of the I/O readiness notification.
	*runtime·netpolluser(pd) = 0;
	r = runtime·port_associate(portfd, PORT_SOURCE_FD, fd, 0, (uintptr)pd);
	runtime·netpollunlock(pd);
	return r;
}

int32
runtime·netpollclose(uintptr fd)
{
	return runtime·port_dissociate(portfd, PORT_SOURCE_FD, fd);
}

// Updates the association with a new set of interested events. After
// this call, port_getn will return one and only one event for that
// particular descriptor, so this function needs to be called again.
void
runtime·netpollupdate(PollDesc* pd, uint32 set, uint32 clear)
{
	uint32 *ep, old, events;
	uintptr fd = runtime·netpollfd(pd);
	ep = (uint32*)runtime·netpolluser(pd);

	if(runtime·netpollclosing(pd))
		return;

	old = *ep;
	events = (old & ~clear) | set;
	if(old == events)
		return;

	if(events && runtime·port_associate(portfd, PORT_SOURCE_FD, fd, events, (uintptr)pd) != 0) {
		runtime·printf("netpollupdate: failed to associate (%d)\n", errno);
		runtime·throw("netpollupdate: failed to associate");
	} 
	*ep = events;
}

// subscribe the fd to the port such that port_getn will return one event.
void
runtime·netpollarm(PollDesc* pd, int32 mode)
{
	runtime·netpolllock(pd);
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
	runtime·netpollunlock(pd);
}

// polls for ready network connections
// returns list of goroutines that become runnable
G*
runtime·netpoll(bool block)
{
	static int32 lasterr;
	PortEvent events[128], *ev;
	PollDesc *pd;
	int32 i, mode, clear;
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
			runtime·printf("runtime: port_getn on fd %d failed with %d\n", portfd, errno);
		}
		goto retry;
	}

	gp = nil;
	for(i = 0; i < n; i++) {
		ev = &events[i];

		if(ev->portev_events == 0)
			continue;
		pd = (PollDesc *)ev->portev_user;

		mode = 0;
		clear = 0;
		if(ev->portev_events & (POLLIN|POLLHUP|POLLERR)) {
			mode += 'r';
			clear |= POLLIN;
		}
		if(ev->portev_events & (POLLOUT|POLLHUP|POLLERR)) {
			mode += 'w';
			clear |= POLLOUT;
		}
		// To effect edge-triggered events, we need to be sure to
		// update our association with whatever events were not
		// set with the event. For example if we are registered
		// for POLLIN|POLLOUT, and we get POLLIN, besides waking
		// the goroutine interested in POLLIN we have to not forget
		// about the one interested in POLLOUT.
		if(clear != 0) {
			runtime·netpolllock(pd);
			runtime·netpollupdate(pd, 0, clear);
			runtime·netpollunlock(pd);
		}

		if(mode)
			runtime·netpollready(&gp, pd, mode);
	}

	if(block && gp == nil)
		goto retry;
	return gp;
}

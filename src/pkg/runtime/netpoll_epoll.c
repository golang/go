// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"

int32	runtime·epollcreate(int32 size);
int32	runtime·epollcreate1(int32 flags);
int32	runtime·epollctl(int32 epfd, int32 op, int32 fd, EpollEvent *ev);
int32	runtime·epollwait(int32 epfd, EpollEvent *ev, int32 nev, int32 timeout);
void	runtime·closeonexec(int32 fd);

static int32 epfd = -1;  // epoll descriptor

void
runtime·netpollinit(void)
{
	epfd = runtime·epollcreate1(EPOLL_CLOEXEC);
	if(epfd >= 0)
		return;
	epfd = runtime·epollcreate(1024);
	if(epfd >= 0) {
		runtime·closeonexec(epfd);
		return;
	}
	runtime·printf("netpollinit: failed to create descriptor (%d)\n", -epfd);
	runtime·throw("netpollinit: failed to create descriptor");
}

int32
runtime·netpollopen(int32 fd, PollDesc *pd)
{
	EpollEvent ev;
	int32 res;

	ev.events = EPOLLIN|EPOLLOUT|EPOLLRDHUP|EPOLLET;
	ev.data = (uint64)pd;
	res = runtime·epollctl(epfd, EPOLL_CTL_ADD, fd, &ev);
	return -res;
}

int32
runtime·netpollclose(int32 fd)
{
	EpollEvent ev;
	int32 res;

	res = runtime·epollctl(epfd, EPOLL_CTL_DEL, fd, &ev);
	return -res;
}

// polls for ready network connections
// returns list of goroutines that become runnable
G*
runtime·netpoll(bool block)
{
	static int32 lasterr;
	EpollEvent events[128], *ev;
	int32 n, i, waitms, mode;
	G *gp;

	if(epfd == -1)
		return nil;
	waitms = -1;
	if(!block)
		waitms = 0;
retry:
	n = runtime·epollwait(epfd, events, nelem(events), waitms);
	if(n < 0) {
		if(n != -EINTR && n != lasterr) {
			lasterr = n;
			runtime·printf("runtime: epollwait on fd %d failed with %d\n", epfd, -n);
		}
		goto retry;
	}
	gp = nil;
	for(i = 0; i < n; i++) {
		ev = &events[i];
		if(ev->events == 0)
			continue;
		mode = 0;
		if(ev->events & (EPOLLIN|EPOLLRDHUP|EPOLLHUP|EPOLLERR))
			mode += 'r';
		if(ev->events & (EPOLLOUT|EPOLLHUP|EPOLLERR))
			mode += 'w';
		if(mode)
			runtime·netpollready(&gp, (void*)ev->data, mode);
	}
	if(block && gp == nil)
		goto retry;
	return gp;
}

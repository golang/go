// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"

#define DWORD_MAX 0xffffffff

#pragma dynimport runtime·CreateIoCompletionPort CreateIoCompletionPort "kernel32.dll"
#pragma dynimport runtime·GetQueuedCompletionStatus GetQueuedCompletionStatus "kernel32.dll"
#pragma dynimport runtime·WSAGetOverlappedResult WSAGetOverlappedResult "ws2_32.dll"

extern void *runtime·CreateIoCompletionPort;
extern void *runtime·GetQueuedCompletionStatus;
extern void *runtime·WSAGetOverlappedResult;

#define INVALID_HANDLE_VALUE ((uintptr)-1)

// net_op must be the same as beginning of net.operation. Keep these in sync.
typedef struct net_op net_op;
struct net_op
{
	// used by windows
	Overlapped	o;
	// used by netpoll
	PollDesc*	pd;
	int32	mode;
	int32	errno;
	uint32	qty;
};

typedef struct OverlappedEntry OverlappedEntry;
struct OverlappedEntry
{
	uintptr	key;
	net_op*	op;  // In reality it's Overlapped*, but we cast it to net_op* anyway.
	uintptr	internal;
	uint32	qty;
};

static void handlecompletion(G **gpp, net_op *o, int32 errno, uint32 qty);

static uintptr iocphandle = INVALID_HANDLE_VALUE;  // completion port io handle

void
runtime·netpollinit(void)
{
	iocphandle = (uintptr)runtime·stdcall4(runtime·CreateIoCompletionPort, INVALID_HANDLE_VALUE, 0, 0, DWORD_MAX);
	if(iocphandle == 0) {
		runtime·printf("netpoll: failed to create iocp handle (errno=%d)\n", runtime·getlasterror());
		runtime·throw("netpoll: failed to create iocp handle");
	}
	return;
}

int32
runtime·netpollopen(uintptr fd, PollDesc *pd)
{
	USED(pd);
	if(runtime·stdcall4(runtime·CreateIoCompletionPort, fd, iocphandle, 0, 0) == 0)
		return -runtime·getlasterror();
	return 0;
}

int32
runtime·netpollclose(uintptr fd)
{
	// nothing to do
	USED(fd);
	return 0;
}

void
runtime·netpollarm(PollDesc* pd, int32 mode)
{
	USED(pd, mode);
	runtime·throw("unused");
}

// Polls for completed network IO.
// Returns list of goroutines that become runnable.
G*
runtime·netpoll(bool block)
{
	OverlappedEntry entries[64];
	uint32 wait, qty, key, flags, n, i;
	int32 errno;
	net_op *op;
	G *gp;

	if(iocphandle == INVALID_HANDLE_VALUE)
		return nil;
	gp = nil;
	wait = 0;
	if(block)
		wait = INFINITE;
retry:
	if(runtime·GetQueuedCompletionStatusEx != nil) {
		n = nelem(entries) / runtime·gomaxprocs;
		if(n < 8)
			n = 8;
		if(block)
			g->m->blocked = true;
		if(runtime·stdcall6(runtime·GetQueuedCompletionStatusEx, iocphandle, (uintptr)entries, n, (uintptr)&n, wait, 0) == 0) {
			g->m->blocked = false;
			errno = runtime·getlasterror();
			if(!block && errno == WAIT_TIMEOUT)
				return nil;
			runtime·printf("netpoll: GetQueuedCompletionStatusEx failed (errno=%d)\n", errno);
			runtime·throw("netpoll: GetQueuedCompletionStatusEx failed");
		}
		g->m->blocked = false;
		for(i = 0; i < n; i++) {
			op = entries[i].op;
			errno = 0;
			qty = 0;
			if(runtime·stdcall5(runtime·WSAGetOverlappedResult, runtime·netpollfd(op->pd), (uintptr)op, (uintptr)&qty, 0, (uintptr)&flags) == 0)
				errno = runtime·getlasterror();
			handlecompletion(&gp, op, errno, qty);
		}
	} else {
		op = nil;
		errno = 0;
		qty = 0;
		if(block)
			g->m->blocked = true;
		if(runtime·stdcall5(runtime·GetQueuedCompletionStatus, iocphandle, (uintptr)&qty, (uintptr)&key, (uintptr)&op, wait) == 0) {
			g->m->blocked = false;
			errno = runtime·getlasterror();
			if(!block && errno == WAIT_TIMEOUT)
				return nil;
			if(op == nil) {
				runtime·printf("netpoll: GetQueuedCompletionStatus failed (errno=%d)\n", errno);
				runtime·throw("netpoll: GetQueuedCompletionStatus failed");
			}
			// dequeued failed IO packet, so report that
		}
		g->m->blocked = false;
		handlecompletion(&gp, op, errno, qty);
	}
	if(block && gp == nil)
		goto retry;
	return gp;
}

static void
handlecompletion(G **gpp, net_op *op, int32 errno, uint32 qty)
{
	int32 mode;

	if(op == nil)
		runtime·throw("netpoll: GetQueuedCompletionStatus returned op == nil");
	mode = op->mode;
	if(mode != 'r' && mode != 'w') {
		runtime·printf("netpoll: GetQueuedCompletionStatus returned invalid mode=%d\n", mode);
		runtime·throw("netpoll: GetQueuedCompletionStatus returned invalid mode");
	}
	op->errno = errno;
	op->qty = qty;
	runtime·netpollready(gpp, op->pd, mode);
}

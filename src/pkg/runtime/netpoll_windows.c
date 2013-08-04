// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"

#define DWORD_MAX 0xffffffff

#pragma dynimport runtime·CreateIoCompletionPort CreateIoCompletionPort "kernel32.dll"
#pragma dynimport runtime·GetQueuedCompletionStatus GetQueuedCompletionStatus "kernel32.dll"

extern void *runtime·CreateIoCompletionPort;
extern void *runtime·GetQueuedCompletionStatus;

#define INVALID_HANDLE_VALUE ((uintptr)-1)

// net_anOp must be the same as beginning of net.anOp. Keep these in sync.
typedef struct net_anOp net_anOp;
struct net_anOp
{
	// used by windows
	Overlapped	o;
	// used by netpoll
	uintptr	runtimeCtx;
	int32	mode;
	int32	errno;
	uint32	qty;
};

static uintptr iocphandle = INVALID_HANDLE_VALUE;  // completion port io handle

void
runtime·netpollinit(void)
{
	iocphandle = (uintptr)runtime·stdcall(runtime·CreateIoCompletionPort, 4, INVALID_HANDLE_VALUE, (uintptr)0, (uintptr)0, (uintptr)DWORD_MAX);
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
	if(runtime·stdcall(runtime·CreateIoCompletionPort, 4, fd, iocphandle, (uintptr)0, (uintptr)0) == 0)
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

// Polls for completed network IO.
// Returns list of goroutines that become runnable.
G*
runtime·netpoll(bool block)
{
	uint32 wait, qty, key;
	int32 mode, errno;
	net_anOp *o;
	G *gp;

	if(iocphandle == INVALID_HANDLE_VALUE)
		return nil;
	gp = nil;
retry:
	o = nil;
	errno = 0;
	qty = 0;
	wait = INFINITE;
	if(!block)
		wait = 0;
	// TODO(brainman): Need a loop here to fetch all pending notifications
	// (or at least a batch). Scheduler will behave better if is given
	// a batch of newly runnable goroutines.
	// TODO(brainman): Call GetQueuedCompletionStatusEx() here when possible.
	if(runtime·stdcall(runtime·GetQueuedCompletionStatus, 5, iocphandle, &qty, &key, &o, (uintptr)wait) == 0) {
		errno = runtime·getlasterror();
		if(o == nil && errno == WAIT_TIMEOUT) {
			if(!block)
				return nil;
			runtime·throw("netpoll: GetQueuedCompletionStatus timed out");
		}
		if(o == nil) {
			runtime·printf("netpoll: GetQueuedCompletionStatus failed (errno=%d)\n", errno);
			runtime·throw("netpoll: GetQueuedCompletionStatus failed");
		}
		// dequeued failed IO packet, so report that
	}
	if(o == nil)
		runtime·throw("netpoll: GetQueuedCompletionStatus returned o == nil");
	mode = o->mode;
	if(mode != 'r' && mode != 'w') {
		runtime·printf("netpoll: GetQueuedCompletionStatus returned invalid mode=%d\n", mode);
		runtime·throw("netpoll: GetQueuedCompletionStatus returned invalid mode");
	}
	o->errno = errno;
	o->qty = qty;
	runtime·netpollready(&gp, (void*)o->runtimeCtx, mode);
	if(block && gp == nil)
		goto retry;
	return gp;
}

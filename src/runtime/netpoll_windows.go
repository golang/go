// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/atomic"
	"unsafe"
)

const _DWORD_MAX = 0xffffffff

const _INVALID_HANDLE_VALUE = ^uintptr(0)

// net_op must be the same as beginning of internal/poll.operation.
// Keep these in sync.
type net_op struct {
	// used by windows
	o overlapped
	// used by netpoll
	pd    *pollDesc
	mode  int32
	errno int32
	qty   uint32
}

type overlappedEntry struct {
	key      *pollDesc
	op       *net_op // In reality it's *overlapped, but we cast it to *net_op anyway.
	internal uintptr
	qty      uint32
}

var (
	iocphandle uintptr = _INVALID_HANDLE_VALUE // completion port io handle

	netpollWakeSig atomic.Uint32 // used to avoid duplicate calls of netpollBreak
)

func netpollinit() {
	iocphandle = stdcall4(_CreateIoCompletionPort, _INVALID_HANDLE_VALUE, 0, 0, _DWORD_MAX)
	if iocphandle == 0 {
		println("runtime: CreateIoCompletionPort failed (errno=", getlasterror(), ")")
		throw("runtime: netpollinit failed")
	}
}

func netpollIsPollDescriptor(fd uintptr) bool {
	return fd == iocphandle
}

func netpollopen(fd uintptr, pd *pollDesc) int32 {
	// TODO(iant): Consider using taggedPointer on 64-bit systems.
	if stdcall4(_CreateIoCompletionPort, fd, iocphandle, uintptr(unsafe.Pointer(pd)), 0) == 0 {
		return int32(getlasterror())
	}
	return 0
}

func netpollclose(fd uintptr) int32 {
	// nothing to do
	return 0
}

func netpollarm(pd *pollDesc, mode int) {
	throw("runtime: unused")
}

func netpollBreak() {
	// Failing to cas indicates there is an in-flight wakeup, so we're done here.
	if !netpollWakeSig.CompareAndSwap(0, 1) {
		return
	}

	if stdcall4(_PostQueuedCompletionStatus, iocphandle, 0, 0, 0) == 0 {
		println("runtime: netpoll: PostQueuedCompletionStatus failed (errno=", getlasterror(), ")")
		throw("runtime: netpoll: PostQueuedCompletionStatus failed")
	}
}

// netpoll checks for ready network connections.
// Returns list of goroutines that become runnable.
// delay < 0: blocks indefinitely
// delay == 0: does not block, just polls
// delay > 0: block for up to that many nanoseconds
func netpoll(delay int64) (gList, int32) {
	var entries [64]overlappedEntry
	var wait, qty, flags, n, i uint32
	var errno int32
	var op *net_op
	var toRun gList

	mp := getg().m

	if iocphandle == _INVALID_HANDLE_VALUE {
		return gList{}, 0
	}
	if delay < 0 {
		wait = _INFINITE
	} else if delay == 0 {
		wait = 0
	} else if delay < 1e6 {
		wait = 1
	} else if delay < 1e15 {
		wait = uint32(delay / 1e6)
	} else {
		// An arbitrary cap on how long to wait for a timer.
		// 1e9 ms == ~11.5 days.
		wait = 1e9
	}

	n = uint32(len(entries) / int(gomaxprocs))
	if n < 8 {
		n = 8
	}
	if delay != 0 {
		mp.blocked = true
	}
	if stdcall6(_GetQueuedCompletionStatusEx, iocphandle, uintptr(unsafe.Pointer(&entries[0])), uintptr(n), uintptr(unsafe.Pointer(&n)), uintptr(wait), 0) == 0 {
		mp.blocked = false
		errno = int32(getlasterror())
		if errno == _WAIT_TIMEOUT {
			return gList{}, 0
		}
		println("runtime: GetQueuedCompletionStatusEx failed (errno=", errno, ")")
		throw("runtime: netpoll failed")
	}
	mp.blocked = false
	delta := int32(0)
	for i = 0; i < n; i++ {
		op = entries[i].op
		if op != nil && op.pd == entries[i].key {
			errno = 0
			qty = 0
			if stdcall5(_WSAGetOverlappedResult, op.pd.fd, uintptr(unsafe.Pointer(op)), uintptr(unsafe.Pointer(&qty)), 0, uintptr(unsafe.Pointer(&flags))) == 0 {
				errno = int32(getlasterror())
			}
			delta += handlecompletion(&toRun, op, errno, qty)
		} else {
			netpollWakeSig.Store(0)
			if delay == 0 {
				// Forward the notification to the
				// blocked poller.
				netpollBreak()
			}
		}
	}
	return toRun, delta
}

func handlecompletion(toRun *gList, op *net_op, errno int32, qty uint32) int32 {
	mode := op.mode
	if mode != 'r' && mode != 'w' {
		println("runtime: GetQueuedCompletionStatusEx returned invalid mode=", mode)
		throw("runtime: netpoll failed")
	}
	op.errno = errno
	op.qty = qty
	return netpollready(toRun, op.pd, mode)
}

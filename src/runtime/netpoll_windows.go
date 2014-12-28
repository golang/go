// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"unsafe"
)

const _DWORD_MAX = 0xffffffff

//go:cgo_import_dynamic runtime._CreateIoCompletionPort CreateIoCompletionPort "kernel32.dll"
//go:cgo_import_dynamic runtime._GetQueuedCompletionStatus GetQueuedCompletionStatus "kernel32.dll"
//go:cgo_import_dynamic runtime._WSAGetOverlappedResult WSAGetOverlappedResult "ws2_32.dll"

var (
	_CreateIoCompletionPort,
	_GetQueuedCompletionStatus,
	_WSAGetOverlappedResult stdFunction
)

const _INVALID_HANDLE_VALUE = ^uintptr(0)

// net_op must be the same as beginning of net.operation. Keep these in sync.
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
	key      uintptr
	op       *net_op // In reality it's *overlapped, but we cast it to *net_op anyway.
	internal uintptr
	qty      uint32
}

var iocphandle uintptr = _INVALID_HANDLE_VALUE // completion port io handle

func netpollinit() {
	iocphandle = uintptr(stdcall4(_CreateIoCompletionPort, _INVALID_HANDLE_VALUE, 0, 0, _DWORD_MAX))
	if iocphandle == 0 {
		println("netpoll: failed to create iocp handle (errno=", getlasterror(), ")")
		throw("netpoll: failed to create iocp handle")
	}
}

func netpollopen(fd uintptr, pd *pollDesc) int32 {
	if stdcall4(_CreateIoCompletionPort, fd, iocphandle, 0, 0) == 0 {
		return -int32(getlasterror())
	}
	return 0
}

func netpollclose(fd uintptr) int32 {
	// nothing to do
	return 0
}

func netpollarm(pd *pollDesc, mode int) {
	throw("unused")
}

// Polls for completed network IO.
// Returns list of goroutines that become runnable.
func netpoll(block bool) *g {
	var entries [64]overlappedEntry
	var wait, qty, key, flags, n, i uint32
	var errno int32
	var op *net_op
	var gp *g

	mp := getg().m

	if iocphandle == _INVALID_HANDLE_VALUE {
		return nil
	}
	gp = nil
	wait = 0
	if block {
		wait = _INFINITE
	}
retry:
	if _GetQueuedCompletionStatusEx != nil {
		n = uint32(len(entries) / int(gomaxprocs))
		if n < 8 {
			n = 8
		}
		if block {
			mp.blocked = true
		}
		if stdcall6(_GetQueuedCompletionStatusEx, iocphandle, uintptr(unsafe.Pointer(&entries[0])), uintptr(n), uintptr(unsafe.Pointer(&n)), uintptr(wait), 0) == 0 {
			mp.blocked = false
			errno = int32(getlasterror())
			if !block && errno == _WAIT_TIMEOUT {
				return nil
			}
			println("netpoll: GetQueuedCompletionStatusEx failed (errno=", errno, ")")
			throw("netpoll: GetQueuedCompletionStatusEx failed")
		}
		mp.blocked = false
		for i = 0; i < n; i++ {
			op = entries[i].op
			errno = 0
			qty = 0
			if stdcall5(_WSAGetOverlappedResult, op.pd.fd, uintptr(unsafe.Pointer(op)), uintptr(unsafe.Pointer(&qty)), 0, uintptr(unsafe.Pointer(&flags))) == 0 {
				errno = int32(getlasterror())
			}
			handlecompletion(&gp, op, errno, qty)
		}
	} else {
		op = nil
		errno = 0
		qty = 0
		if block {
			mp.blocked = true
		}
		if stdcall5(_GetQueuedCompletionStatus, iocphandle, uintptr(unsafe.Pointer(&qty)), uintptr(unsafe.Pointer(&key)), uintptr(unsafe.Pointer(&op)), uintptr(wait)) == 0 {
			mp.blocked = false
			errno = int32(getlasterror())
			if !block && errno == _WAIT_TIMEOUT {
				return nil
			}
			if op == nil {
				println("netpoll: GetQueuedCompletionStatus failed (errno=", errno, ")")
				throw("netpoll: GetQueuedCompletionStatus failed")
			}
			// dequeued failed IO packet, so report that
		}
		mp.blocked = false
		handlecompletion(&gp, op, errno, qty)
	}
	if block && gp == nil {
		goto retry
	}
	return gp
}

func handlecompletion(gpp **g, op *net_op, errno int32, qty uint32) {
	if op == nil {
		throw("netpoll: GetQueuedCompletionStatus returned op == nil")
	}
	mode := op.mode
	if mode != 'r' && mode != 'w' {
		println("netpoll: GetQueuedCompletionStatus returned invalid mode=", mode)
		throw("netpoll: GetQueuedCompletionStatus returned invalid mode")
	}
	op.errno = errno
	op.qty = qty
	netpollready((**g)(noescape(unsafe.Pointer(gpp))), op.pd, mode)
}

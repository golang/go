// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/goarch"
	"runtime/internal/atomic"
	"unsafe"
)

const _DWORD_MAX = 0xffffffff

const _INVALID_HANDLE_VALUE = ^uintptr(0)

// Sources are used to identify the event that created an overlapped entry.
// The source values are arbitrary. There is no risk of collision with user
// defined values because the only way to set the key of an overlapped entry
// is using the iocphandle, which is not accessible to user code.
const (
	netpollSourceReady = iota + 1
	netpollSourceBreak
)

const (
	// sourceBits is the number of bits needed to represent a source.
	// 4 bits can hold 16 different sources, which is more than enough.
	// It is set to a low value so the overlapped entry key can
	// contain as much bits as possible for the pollDesc pointer.
	sourceBits  = 4 // 4 bits can hold 16 different sources, which is more than enough.
	sourceMasks = 1<<sourceBits - 1
)

// packNetpollKey creates a key from a source and a tag.
// Bits that don't fit in the result are discarded.
func packNetpollKey(source uint8, pd *pollDesc) uintptr {
	// TODO: Consider combining the source with pd.fdseq to detect stale pollDescs.
	if source > (1<<sourceBits)-1 {
		// Also fail on 64-bit systems, even though it can hold more bits.
		throw("runtime: source value is too large")
	}
	if goarch.PtrSize == 4 {
		return uintptr(unsafe.Pointer(pd))<<sourceBits | uintptr(source)
	}
	return uintptr(taggedPointerPack(unsafe.Pointer(pd), uintptr(source)))
}

// unpackNetpollSource returns the source packed key.
func unpackNetpollSource(key uintptr) uint8 {
	if goarch.PtrSize == 4 {
		return uint8(key & sourceMasks)
	}
	return uint8(taggedPointer(key).tag())
}

// pollOperation must be the same as beginning of internal/poll.operation.
// Keep these in sync.
type pollOperation struct {
	// used by windows
	_ overlapped
	// used by netpoll
	pd   *pollDesc
	mode int32
}

// pollOperationFromOverlappedEntry returns the pollOperation contained in
// e. It can return nil if the entry is not from internal/poll.
// See go.dev/issue/58870
func pollOperationFromOverlappedEntry(e *overlappedEntry) *pollOperation {
	if e.ov == nil {
		return nil
	}
	op := (*pollOperation)(unsafe.Pointer(e.ov))
	// Check that the key matches the pollDesc pointer.
	var keyMatch bool
	if goarch.PtrSize == 4 {
		keyMatch = e.key&^sourceMasks == uintptr(unsafe.Pointer(op.pd))<<sourceBits
	} else {
		keyMatch = (*pollDesc)(taggedPointer(e.key).pointer()) == op.pd
	}
	if !keyMatch {
		return nil
	}
	return op
}

// overlappedEntry contains the information returned by a call to GetQueuedCompletionStatusEx.
// https://learn.microsoft.com/en-us/windows/win32/api/minwinbase/ns-minwinbase-overlapped_entry
type overlappedEntry struct {
	key      uintptr
	ov       *overlapped
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
	key := packNetpollKey(netpollSourceReady, pd)
	if stdcall4(_CreateIoCompletionPort, fd, iocphandle, key, 0) == 0 {
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

	key := packNetpollKey(netpollSourceBreak, nil)
	if stdcall4(_PostQueuedCompletionStatus, iocphandle, 0, key, 0) == 0 {
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
	var wait, n, i uint32
	var errno int32
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
		e := &entries[i]
		switch unpackNetpollSource(e.key) {
		case netpollSourceReady:
			op := pollOperationFromOverlappedEntry(e)
			if op == nil {
				// Entry from outside the Go runtime and internal/poll, ignore.
				continue
			}
			// Entry from internal/poll.
			mode := op.mode
			if mode != 'r' && mode != 'w' {
				println("runtime: GetQueuedCompletionStatusEx returned net_op with invalid mode=", mode)
				throw("runtime: netpoll failed")
			}
			delta += netpollready(&toRun, op.pd, mode)
		case netpollSourceBreak:
			netpollWakeSig.Store(0)
			if delay == 0 {
				// Forward the notification to the blocked poller.
				netpollBreak()
			}
		default:
			println("runtime: GetQueuedCompletionStatusEx returned net_op with invalid key=", e.key)
			throw("runtime: netpoll failed")
		}
	}
	return toRun, delta
}

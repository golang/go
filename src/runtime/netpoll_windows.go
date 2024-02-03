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

// Sources are used to identify the event that created an overlapped entry.
// The source values are arbitrary. There is no risk of collision with user
// defined values because the only way to set the key of an overlapped entry
// is using the iocphandle, which is not accessible to user code.
const (
	netpollSourceReady = iota + 1
	netpollSourceBreak
)

const (
	netPollKeySourceBits = 8
	// Use 19 bits for the fdseq, which is enough to represent all possible
	// values on 64-bit systems (fdseq is truncated to taggedPointerBits).
	// On 32-bit systems, taggedPointerBits is set to 32 bits, so we are
	// losing precision here, but still have enough entropy to avoid collisions
	// (see netpollopen).
	netPollKeyFDSeqBits = 19
	netPollKeyFDSeqMask = 1<<netPollKeyFDSeqBits - 1
)

// packNetpollKey creates a key from a source and a tag.
// Tag bits that don't fit in the result are discarded.
func packNetpollKey(source uint8, tag uintptr) uintptr {
	return uintptr(source) | tag<<netPollKeySourceBits
}

// unpackNetpollKey returns the source and the tag from a taggedPointer.
func unpackNetpollKey(key uintptr) (source uint8, tag uintptr) {
	return uint8(key), key >> netPollKeySourceBits
}

// net_op must be the same as beginning of internal/poll.operation.
// Keep these in sync.
type net_op struct {
	// used by windows
	_ overlapped
	// used by netpoll
	pd   *pollDesc
	mode int32
}

type overlappedEntry struct {
	key      uintptr
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
	// The tag is used for two purposes:
	// - identify stale pollDescs. See go.dev/issue/59545.
	// - differentiate between entries from internal/poll and entries from
	//   outside the Go runtime, which we want to skip. User code has access
	//   to fd, therefore it can run async operations on it that will end up
	//   adding overlapped entries to our iocp queue. See go.dev/issue/58870.
	//   By setting the tag to the pollDesc's fdseq, the only chance of
	//   collision is if a user creates an overlapped struct with a fdseq that
	//   matches the fdseq of the pollDesc passed to netpollopen, which is quite
	//   unlikely given that fdseq is not exposed to user code.
	tag := pd.fdseq.Load() & netPollKeyFDSeqMask
	key := packNetpollKey(netpollSourceReady, tag)
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

	key := packNetpollKey(netpollSourceBreak, 0)
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
		key, tag := unpackNetpollKey(e.key)
		switch {
		case key == netpollSourceBreak:
			netpollWakeSig.Store(0)
			if delay == 0 {
				// Forward the notification to the blocked poller.
				netpollBreak()
			}
		case key == netpollSourceReady:
			if e.op == nil || e.op.pd == nil || e.op.pd.fdseq.Load()&netPollKeyFDSeqMask != tag&netPollKeyFDSeqMask {
				// Stale entry or entry from outside the Go runtime and internal/poll, ignore.
				// See go.dev/issue/58870.
				continue
			}
			// Entry from internal/poll.
			mode := e.op.mode
			if mode != 'r' && mode != 'w' {
				println("runtime: GetQueuedCompletionStatusEx returned net_op with invalid mode=", mode)
				throw("runtime: netpoll failed")
			}
			delta += netpollready(&toRun, e.op.pd, mode)
		default:
			println("runtime: GetQueuedCompletionStatusEx returned net_op with invalid key=", e.key)
			throw("runtime: netpoll failed")
		}
	}
	return toRun, delta
}

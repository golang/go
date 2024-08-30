// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/goarch"
	"internal/runtime/atomic"
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
	netpollSourceTimer
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
	if iocphandle == _INVALID_HANDLE_VALUE {
		return gList{}, 0
	}

	var entries [64]overlappedEntry
	var wait uint32
	var toRun gList
	mp := getg().m

	if delay >= 1e15 {
		// An arbitrary cap on how long to wait for a timer.
		// 1e15 ns == ~11.5 days.
		delay = 1e15
	}

	if delay > 0 && mp.waitIocpHandle != 0 {
		// GetQueuedCompletionStatusEx doesn't use a high resolution timer internally,
		// so we use a separate higher resolution timer associated with a wait completion
		// packet to wake up the poller. Note that the completion packet can be delivered
		// to another thread, and the Go scheduler expects netpoll to only block up to delay,
		// so we still need to use a timeout with GetQueuedCompletionStatusEx.
		// TODO: Improve the Go scheduler to support non-blocking timers.
		signaled := netpollQueueTimer(delay)
		if signaled {
			// There is a small window between the SetWaitableTimer and the NtAssociateWaitCompletionPacket
			// where the timer can expire. We can return immediately in this case.
			return gList{}, 0
		}
	}
	if delay < 0 {
		wait = _INFINITE
	} else if delay == 0 {
		wait = 0
	} else if delay < 1e6 {
		wait = 1
	} else {
		wait = uint32(delay / 1e6)
	}
	n := len(entries) / int(gomaxprocs)
	if n < 8 {
		n = 8
	}
	if delay != 0 {
		mp.blocked = true
	}
	if stdcall6(_GetQueuedCompletionStatusEx, iocphandle, uintptr(unsafe.Pointer(&entries[0])), uintptr(n), uintptr(unsafe.Pointer(&n)), uintptr(wait), 0) == 0 {
		mp.blocked = false
		errno := getlasterror()
		if errno == _WAIT_TIMEOUT {
			return gList{}, 0
		}
		println("runtime: GetQueuedCompletionStatusEx failed (errno=", errno, ")")
		throw("runtime: netpoll failed")
	}
	mp.blocked = false
	delta := int32(0)
	for i := 0; i < n; i++ {
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
		case netpollSourceTimer:
			// TODO: We could avoid calling NtCancelWaitCompletionPacket for expired wait completion packets.
		default:
			println("runtime: GetQueuedCompletionStatusEx returned net_op with invalid key=", e.key)
			throw("runtime: netpoll failed")
		}
	}
	return toRun, delta
}

// netpollQueueTimer queues a timer to wake up the poller after the given delay.
// It returns true if the timer expired during this call.
func netpollQueueTimer(delay int64) (signaled bool) {
	const (
		STATUS_SUCCESS   = 0x00000000
		STATUS_PENDING   = 0x00000103
		STATUS_CANCELLED = 0xC0000120
	)
	mp := getg().m
	// A wait completion packet can only be associated with one timer at a time,
	// so we need to cancel the previous one if it exists. This wouldn't be necessary
	// if the poller would only be woken up by the timer, in which case the association
	// would be automatically canceled, but it can also be woken up by other events,
	// such as a netpollBreak, so we can get to this point with a timer that hasn't
	// expired yet. In this case, the completion packet can still be picked up by
	// another thread, so defer the cancellation until it is really necessary.
	errno := stdcall2(_NtCancelWaitCompletionPacket, mp.waitIocpHandle, 1)
	switch errno {
	case STATUS_CANCELLED:
		// STATUS_CANCELLED is returned when the associated timer has already expired,
		// in which automatically cancels the wait completion packet.
		fallthrough
	case STATUS_SUCCESS:
		dt := -delay / 100 // relative sleep (negative), 100ns units
		if stdcall6(_SetWaitableTimer, mp.waitIocpTimer, uintptr(unsafe.Pointer(&dt)), 0, 0, 0, 0) == 0 {
			println("runtime: SetWaitableTimer failed; errno=", getlasterror())
			throw("runtime: netpoll failed")
		}
		key := packNetpollKey(netpollSourceTimer, nil)
		if errno := stdcall8(_NtAssociateWaitCompletionPacket, mp.waitIocpHandle, iocphandle, mp.waitIocpTimer, key, 0, 0, 0, uintptr(unsafe.Pointer(&signaled))); errno != 0 {
			println("runtime: NtAssociateWaitCompletionPacket failed; errno=", errno)
			throw("runtime: netpoll failed")
		}
	case STATUS_PENDING:
		// STATUS_PENDING is returned if the wait operation can't be canceled yet.
		// This can happen if this thread was woken up by another event, such as a netpollBreak,
		// and the timer expired just while calling NtCancelWaitCompletionPacket, in which case
		// this call fails to cancel the association to avoid a race condition.
		// This is a rare case, so we can just avoid using the high resolution timer this time.
	default:
		println("runtime: NtCancelWaitCompletionPacket failed; errno=", errno)
		throw("runtime: netpoll failed")
	}
	return signaled
}

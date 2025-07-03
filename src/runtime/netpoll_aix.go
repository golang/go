// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/runtime/atomic"
	"unsafe"
)

// This is based on the former libgo/runtime/netpoll_select.c implementation
// except that it uses poll instead of select and is written in Go.
// It's also based on Solaris implementation for the arming mechanisms

//go:cgo_import_dynamic libc_poll poll "libc.a/shr_64.o"
//go:linkname libc_poll libc_poll

var libc_poll libFunc

//go:nosplit
func poll(pfds *pollfd, npfds uintptr, timeout uintptr) (int32, int32) {
	r, err := syscall3(&libc_poll, uintptr(unsafe.Pointer(pfds)), npfds, timeout)
	return int32(r), int32(err)
}

// pollfd represents the poll structure for AIX operating system.
type pollfd struct {
	fd      int32
	events  int16
	revents int16
}

const _POLLIN = 0x0001
const _POLLOUT = 0x0002
const _POLLHUP = 0x2000
const _POLLERR = 0x4000

var (
	pfds           []pollfd
	pds            []*pollDesc
	mtxpoll        mutex
	mtxset         mutex
	rdwake         int32
	wrwake         int32
	pendingUpdates int32

	netpollWakeSig atomic.Uint32 // used to avoid duplicate calls of netpollBreak
)

func netpollinit() {
	// Create the pipe we use to wakeup poll.
	r, w, errno := nonblockingPipe()
	if errno != 0 {
		throw("netpollinit: failed to create pipe")
	}
	rdwake = r
	wrwake = w

	// Pre-allocate array of pollfd structures for poll.
	pfds = make([]pollfd, 1, 128)

	// Poll the read side of the pipe.
	pfds[0].fd = rdwake
	pfds[0].events = _POLLIN

	pds = make([]*pollDesc, 1, 128)
	pds[0] = nil
}

func netpollIsPollDescriptor(fd uintptr) bool {
	return fd == uintptr(rdwake) || fd == uintptr(wrwake)
}

// netpollwakeup writes on wrwake to wakeup poll before any changes.
func netpollwakeup() {
	if pendingUpdates == 0 {
		pendingUpdates = 1
		b := [1]byte{0}
		write(uintptr(wrwake), unsafe.Pointer(&b[0]), 1)
	}
}

func netpollopen(fd uintptr, pd *pollDesc) int32 {
	lock(&mtxpoll)
	netpollwakeup()

	lock(&mtxset)
	unlock(&mtxpoll)

	// We don't worry about pd.fdseq here,
	// as mtxset protects us from stale pollDescs.

	pd.user = uint32(len(pfds))
	pfds = append(pfds, pollfd{fd: int32(fd)})
	pds = append(pds, pd)
	unlock(&mtxset)
	return 0
}

func netpollclose(fd uintptr) int32 {
	lock(&mtxpoll)
	netpollwakeup()

	lock(&mtxset)
	unlock(&mtxpoll)

	for i := 0; i < len(pfds); i++ {
		if pfds[i].fd == int32(fd) {
			pfds[i] = pfds[len(pfds)-1]
			pfds = pfds[:len(pfds)-1]

			pds[i] = pds[len(pds)-1]
			pds[i].user = uint32(i)
			pds = pds[:len(pds)-1]
			break
		}
	}
	unlock(&mtxset)
	return 0
}

func netpollarm(pd *pollDesc, mode int) {
	lock(&mtxpoll)
	netpollwakeup()

	lock(&mtxset)
	unlock(&mtxpoll)

	switch mode {
	case 'r':
		pfds[pd.user].events |= _POLLIN
	case 'w':
		pfds[pd.user].events |= _POLLOUT
	}
	unlock(&mtxset)
}

// netpollBreak interrupts a poll.
func netpollBreak() {
	// Failing to cas indicates there is an in-flight wakeup, so we're done here.
	if !netpollWakeSig.CompareAndSwap(0, 1) {
		return
	}

	b := [1]byte{0}
	write(uintptr(wrwake), unsafe.Pointer(&b[0]), 1)
}

// netpoll checks for ready network connections.
// Returns a list of goroutines that become runnable,
// and a delta to add to netpollWaiters.
// This must never return an empty list with a non-zero delta.
//
// delay < 0: blocks indefinitely
// delay == 0: does not block, just polls
// delay > 0: block for up to that many nanoseconds
//
//go:nowritebarrierrec
func netpoll(delay int64) (gList, int32) {
	var timeout uintptr
	if delay < 0 {
		timeout = ^uintptr(0)
	} else if delay == 0 {
		// TODO: call poll with timeout == 0
		return gList{}, 0
	} else if delay < 1e6 {
		timeout = 1
	} else if delay < 1e15 {
		timeout = uintptr(delay / 1e6)
	} else {
		// An arbitrary cap on how long to wait for a timer.
		// 1e9 ms == ~11.5 days.
		timeout = 1e9
	}
retry:
	lock(&mtxpoll)
	lock(&mtxset)
	pendingUpdates = 0
	unlock(&mtxpoll)

	n, e := poll(&pfds[0], uintptr(len(pfds)), timeout)
	if n < 0 {
		if e != _EINTR {
			println("errno=", e, " len(pfds)=", len(pfds))
			throw("poll failed")
		}
		unlock(&mtxset)
		// If a timed sleep was interrupted, just return to
		// recalculate how long we should sleep now.
		if timeout > 0 {
			return gList{}, 0
		}
		goto retry
	}
	// Check if some descriptors need to be changed
	if n != 0 && pfds[0].revents&(_POLLIN|_POLLHUP|_POLLERR) != 0 {
		if delay != 0 {
			// A netpollwakeup could be picked up by a
			// non-blocking poll. Only clear the wakeup
			// if blocking.
			var b [1]byte
			for read(rdwake, unsafe.Pointer(&b[0]), 1) == 1 {
			}
			netpollWakeSig.Store(0)
		}
		// Still look at the other fds even if the mode may have
		// changed, as netpollBreak might have been called.
		n--
	}
	var toRun gList
	delta := int32(0)
	for i := 1; i < len(pfds) && n > 0; i++ {
		pfd := &pfds[i]

		var mode int32
		if pfd.revents&(_POLLIN|_POLLHUP|_POLLERR) != 0 {
			mode += 'r'
			pfd.events &= ^_POLLIN
		}
		if pfd.revents&(_POLLOUT|_POLLHUP|_POLLERR) != 0 {
			mode += 'w'
			pfd.events &= ^_POLLOUT
		}
		if mode != 0 {
			pds[i].setEventErr(pfd.revents == _POLLERR, 0)
			delta += netpollready(&toRun, pds[i], mode)
			n--
		}
	}
	unlock(&mtxset)
	return toRun, delta
}

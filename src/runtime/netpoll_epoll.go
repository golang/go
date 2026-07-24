// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux

package runtime

import (
	"internal/runtime/atomic"
	"internal/runtime/syscall/linux"
	"unsafe"
)

var (
	epfd             int32         = -1 // epoll descriptor
	netpollEventFd   uintptr            // eventfd for netpollBreak
	netpollWakeSig   atomic.Uint32      // used to avoid duplicate calls of netpollBreak
	epollpwait2Avail bool               // true if epoll_pwait2 is available (Linux 5.11+)
)

// netpollEpollPwait2Init decides whether epoll_pwait2 is available.
// Opt-in via GODEBUG=epollpwait2=1: precise per-timer wakeups can
// measurably increase wakeup frequency and CPU usage for workloads
// with many similar short-lived timers. Must run after parsedebugvars.
//
// Check the kernel version first. Even on a new-enough kernel, probe
// the syscall since seccomp may block it. The probe expects EBADF from
// an invalid epfd, treating any other result as unavailable.
func netpollEpollPwait2Init() {
	if debug.epollpwait2 == 0 {
		return
	}
	if kv, ok := getKernelVersion(); ok && !kv.GE(5, 11) {
		return
	}
	_, errno := linux.EpollPwait2(-1, nil, 0, nil)
	const badf = 9 // EBADF
	epollpwait2Avail = errno == badf
}

func netpollinit() {
	var errno uintptr
	epfd, errno = linux.EpollCreate1(linux.EPOLL_CLOEXEC)
	if errno != 0 {
		println("runtime: epollcreate failed with", errno)
		throw("runtime: netpollinit failed")
	}
	efd, errno := linux.Eventfd(0, linux.EFD_CLOEXEC|linux.EFD_NONBLOCK)
	if errno != 0 {
		println("runtime: eventfd failed with", errno)
		throw("runtime: eventfd failed")
	}
	ev := linux.EpollEvent{
		Events: linux.EPOLLIN,
	}
	*(**uintptr)(unsafe.Pointer(&ev.Data)) = &netpollEventFd
	errno = linux.EpollCtl(epfd, linux.EPOLL_CTL_ADD, efd, &ev)
	if errno != 0 {
		println("runtime: epollctl failed with", errno)
		throw("runtime: epollctl failed")
	}
	netpollEventFd = uintptr(efd)
}

func netpollIsPollDescriptor(fd uintptr) bool {
	return fd == uintptr(epfd) || fd == netpollEventFd
}

func netpollopen(fd uintptr, pd *pollDesc) uintptr {
	var ev linux.EpollEvent
	ev.Events = linux.EPOLLIN | linux.EPOLLOUT | linux.EPOLLRDHUP | linux.EPOLLET
	tp := taggedPointerPack(unsafe.Pointer(pd), pd.fdseq.Load())
	*(*taggedPointer)(unsafe.Pointer(&ev.Data)) = tp
	return linux.EpollCtl(epfd, linux.EPOLL_CTL_ADD, int32(fd), &ev)
}

func netpollclose(fd uintptr) uintptr {
	var ev linux.EpollEvent
	return linux.EpollCtl(epfd, linux.EPOLL_CTL_DEL, int32(fd), &ev)
}

func netpollarm(pd *pollDesc, mode int) {
	throw("runtime: unused")
}

// netpollBreak interrupts an epollwait.
func netpollBreak() {
	// Failing to cas indicates there is an in-flight wakeup, so we're done here.
	if !netpollWakeSig.CompareAndSwap(0, 1) {
		return
	}

	var one uint64 = 1
	oneSize := int32(unsafe.Sizeof(one))
	for {
		n := write(netpollEventFd, noescape(unsafe.Pointer(&one)), oneSize)
		if n == oneSize {
			break
		}
		if n == -_EINTR {
			continue
		}
		if n == -_EAGAIN {
			return
		}
		println("runtime: netpollBreak write failed with", -n)
		throw("runtime: netpollBreak write failed")
	}
}

// netpollCoalesceDelay rounds delay up to the nearest coalescing bucket.
// epoll_wait's 1ms floor implicitly batches timers with nearby deadlines
// into a single wakeup; epoll_pwait2 does not. Rounding up to a bucket
// boundary recovers that batching: timers within the same bucket share a
// wakeup. Delay is always rounded up, never down, so no timer fires early.
func netpollCoalesceDelay(delay int64) int64 {
	switch {
	case delay < 100_000: // <100µs: 1µs buckets
		return ((delay + 999) / 1_000) * 1_000
	case delay < 1_000_000: // <1ms: 10µs buckets
		return ((delay + 9_999) / 10_000) * 10_000
	case delay < 10_000_000: // <10ms: 100µs buckets
		return ((delay + 99_999) / 100_000) * 100_000
	default: // ≥10ms: 1ms buckets, same as epoll_wait
		return ((delay + 999_999) / 1_000_000) * 1_000_000
	}
}

// netpoll checks for ready network connections.
// Returns a list of goroutines that become runnable,
// and a delta to add to netpollWaiters.
// This must never return an empty list with a non-zero delta.
//
// delay < 0: blocks indefinitely
// delay == 0: does not block, just polls
// delay > 0: block for up to that many nanoseconds
func netpoll(delay int64) (gList, int32) {
	if epfd == -1 {
		return gList{}, 0
	}
	var events [128]linux.EpollEvent
	var n int32
	var errno uintptr
retry:
	if epollpwait2Avail && delay != 0 {
		// Use epoll_pwait2 for nanosecond-precision timeouts (Linux 5.11+).
		// This eliminates the 1ms rounding applied by epoll_wait, improving
		// deadline accuracy for sub-millisecond timeouts. Apply graduated
		// coalescing to recover the wakeup batching that epoll_wait's 1ms
		// floor provided: nearby timers round into the same bucket and share
		// a single wakeup, bounding the per-timer wakeup regression.
		var ts *linux.KernelTimespec
		if delay > 0 {
			var timeout linux.KernelTimespec
			timeout.SetNsec(netpollCoalesceDelay(delay))
			ts = &timeout
		}
		// delay < 0: ts == nil, blocks indefinitely.
		n, errno = linux.EpollPwait2(epfd, events[:], int32(len(events)), ts)
	} else {
		// epoll_pwait2 unavailable or delay == 0 (non-blocking): use epoll_wait.
		// Convert delay to milliseconds for epoll_wait.
		var waitms int32
		if delay < 0 {
			waitms = -1
		} else if delay == 0 {
			waitms = 0
		} else if delay < 1e6 {
			waitms = 1
		} else if delay < 1e15 {
			waitms = int32(delay / 1e6)
		} else {
			// An arbitrary cap on how long to wait for a timer.
			// 1e9 ms == ~11.5 days.
			waitms = 1e9
		}
		n, errno = linux.EpollWait(epfd, events[:], int32(len(events)), waitms)
	}
	if errno != 0 {
		if errno != _EINTR {
			println("runtime: epollwait on fd", epfd, "failed with", errno)
			throw("runtime: netpoll failed")
		}
		// If a timed sleep was interrupted, just return to
		// recalculate how long we should sleep now.
		if delay > 0 {
			return gList{}, 0
		}
		goto retry
	}
	var toRun gList
	delta := int32(0)
	for i := int32(0); i < n; i++ {
		ev := events[i]
		if ev.Events == 0 {
			continue
		}

		if *(**uintptr)(unsafe.Pointer(&ev.Data)) == &netpollEventFd {
			if ev.Events != linux.EPOLLIN {
				println("runtime: netpoll: eventfd ready for", ev.Events)
				throw("runtime: netpoll: eventfd ready for something unexpected")
			}
			if delay != 0 {
				// netpollBreak could be picked up by a
				// nonblocking poll. Only read the 8-byte
				// integer if blocking.
				// Since EFD_SEMAPHORE was not specified,
				// the eventfd counter will be reset to 0.
				var one uint64
				read(int32(netpollEventFd), noescape(unsafe.Pointer(&one)), int32(unsafe.Sizeof(one)))
				netpollWakeSig.Store(0)
			}
			continue
		}

		var mode int32
		if ev.Events&(linux.EPOLLIN|linux.EPOLLRDHUP|linux.EPOLLHUP|linux.EPOLLERR) != 0 {
			mode += 'r'
		}
		if ev.Events&(linux.EPOLLOUT|linux.EPOLLHUP|linux.EPOLLERR) != 0 {
			mode += 'w'
		}
		if mode != 0 {
			tp := *(*taggedPointer)(unsafe.Pointer(&ev.Data))
			pd := (*pollDesc)(tp.pointer())
			tag := tp.tag()
			if pd.fdseq.Load() == tag {
				pd.setEventErr(ev.Events == linux.EPOLLERR, tag)
				delta += netpollready(&toRun, pd, mode)
			}
		}
	}
	return toRun, delta
}

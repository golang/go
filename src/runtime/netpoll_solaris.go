// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/goarch"
	"runtime/internal/atomic"
	"unsafe"
)

// Solaris runtime-integrated network poller.
//
// Solaris uses event ports for scalable network I/O. Event
// ports are level-triggered, unlike epoll and kqueue which
// can be configured in both level-triggered and edge-triggered
// mode. Level triggering means we have to keep track of a few things
// ourselves. After we receive an event for a file descriptor,
// it's our responsibility to ask again to be notified for future
// events for that descriptor. When doing this we must keep track of
// what kind of events the goroutines are currently interested in,
// for example a fd may be open both for reading and writing.
//
// A description of the high level operation of this code
// follows. Networking code will get a file descriptor by some means
// and will register it with the netpolling mechanism by a code path
// that eventually calls runtime·netpollopen. runtime·netpollopen
// calls port_associate with an empty event set. That means that we
// will not receive any events at this point. The association needs
// to be done at this early point because we need to process the I/O
// readiness notification at some point in the future. If I/O becomes
// ready when nobody is listening, when we finally care about it,
// nobody will tell us anymore.
//
// Beside calling runtime·netpollopen, the networking code paths
// will call runtime·netpollarm each time goroutines are interested
// in doing network I/O. Because now we know what kind of I/O we
// are interested in (reading/writing), we can call port_associate
// passing the correct type of event set (POLLIN/POLLOUT). As we made
// sure to have already associated the file descriptor with the port,
// when we now call port_associate, we will unblock the main poller
// loop (in runtime·netpoll) right away if the socket is actually
// ready for I/O.
//
// The main poller loop runs in its own thread waiting for events
// using port_getn. When an event happens, it will tell the scheduler
// about it using runtime·netpollready. Besides doing this, it must
// also re-associate the events that were not part of this current
// notification with the file descriptor. Failing to do this would
// mean each notification will prevent concurrent code using the
// same file descriptor in parallel.
//
// The logic dealing with re-associations is encapsulated in
// runtime·netpollupdate. This function takes care to associate the
// descriptor only with the subset of events that were previously
// part of the association, except the one that just happened. We
// can't re-associate with that right away, because event ports
// are level triggered so it would cause a busy loop. Instead, that
// association is effected only by the runtime·netpollarm code path,
// when Go code actually asks for I/O.
//
// The open and arming mechanisms are serialized using the lock
// inside PollDesc. This is required because the netpoll loop runs
// asynchronously in respect to other Go code and by the time we get
// to call port_associate to update the association in the loop, the
// file descriptor might have been closed and reopened already. The
// lock allows runtime·netpollupdate to be called synchronously from
// the loop thread while preventing other threads operating to the
// same PollDesc, so once we unblock in the main loop, until we loop
// again we know for sure we are always talking about the same file
// descriptor and can safely access the data we want (the event set).

//go:cgo_import_dynamic libc_port_create port_create "libc.so"
//go:cgo_import_dynamic libc_port_associate port_associate "libc.so"
//go:cgo_import_dynamic libc_port_dissociate port_dissociate "libc.so"
//go:cgo_import_dynamic libc_port_getn port_getn "libc.so"
//go:cgo_import_dynamic libc_port_alert port_alert "libc.so"

//go:linkname libc_port_create libc_port_create
//go:linkname libc_port_associate libc_port_associate
//go:linkname libc_port_dissociate libc_port_dissociate
//go:linkname libc_port_getn libc_port_getn
//go:linkname libc_port_alert libc_port_alert

var (
	libc_port_create,
	libc_port_associate,
	libc_port_dissociate,
	libc_port_getn,
	libc_port_alert libcFunc
	netpollWakeSig atomic.Uint32 // used to avoid duplicate calls of netpollBreak
)

func errno() int32 {
	return *getg().m.perrno
}

func port_create() int32 {
	return int32(sysvicall0(&libc_port_create))
}

func port_associate(port, source int32, object uintptr, events uint32, user uintptr) int32 {
	return int32(sysvicall5(&libc_port_associate, uintptr(port), uintptr(source), object, uintptr(events), user))
}

func port_dissociate(port, source int32, object uintptr) int32 {
	return int32(sysvicall3(&libc_port_dissociate, uintptr(port), uintptr(source), object))
}

func port_getn(port int32, evs *portevent, max uint32, nget *uint32, timeout *timespec) int32 {
	return int32(sysvicall5(&libc_port_getn, uintptr(port), uintptr(unsafe.Pointer(evs)), uintptr(max), uintptr(unsafe.Pointer(nget)), uintptr(unsafe.Pointer(timeout))))
}

func port_alert(port int32, flags, events uint32, user uintptr) int32 {
	return int32(sysvicall4(&libc_port_alert, uintptr(port), uintptr(flags), uintptr(events), user))
}

var portfd int32 = -1

func netpollinit() {
	portfd = port_create()
	if portfd >= 0 {
		closeonexec(portfd)
		return
	}

	print("runtime: port_create failed (errno=", errno(), ")\n")
	throw("runtime: netpollinit failed")
}

func netpollIsPollDescriptor(fd uintptr) bool {
	return fd == uintptr(portfd)
}

func netpollopen(fd uintptr, pd *pollDesc) int32 {
	lock(&pd.lock)
	// We don't register for any specific type of events yet, that's
	// netpollarm's job. We merely ensure we call port_associate before
	// asynchronous connect/accept completes, so when we actually want
	// to do any I/O, the call to port_associate (from netpollarm,
	// with the interested event set) will unblock port_getn right away
	// because of the I/O readiness notification.
	pd.user = 0
	tp := taggedPointerPack(unsafe.Pointer(pd), pd.fdseq.Load())
	// Note that this won't work on a 32-bit system,
	// as taggedPointer is always 64-bits but uintptr will be 32 bits.
	// Fortunately we only support Solaris on amd64.
	if goarch.PtrSize != 8 {
		throw("runtime: netpollopen: unsupported pointer size")
	}
	r := port_associate(portfd, _PORT_SOURCE_FD, fd, 0, uintptr(tp))
	unlock(&pd.lock)
	return r
}

func netpollclose(fd uintptr) int32 {
	return port_dissociate(portfd, _PORT_SOURCE_FD, fd)
}

// Updates the association with a new set of interested events. After
// this call, port_getn will return one and only one event for that
// particular descriptor, so this function needs to be called again.
func netpollupdate(pd *pollDesc, set, clear uint32) {
	if pd.info().closing() {
		return
	}

	old := pd.user
	events := (old & ^clear) | set
	if old == events {
		return
	}

	tp := taggedPointerPack(unsafe.Pointer(pd), pd.fdseq.Load())
	if events != 0 && port_associate(portfd, _PORT_SOURCE_FD, pd.fd, events, uintptr(tp)) != 0 {
		print("runtime: port_associate failed (errno=", errno(), ")\n")
		throw("runtime: netpollupdate failed")
	}
	pd.user = events
}

// subscribe the fd to the port such that port_getn will return one event.
func netpollarm(pd *pollDesc, mode int) {
	lock(&pd.lock)
	switch mode {
	case 'r':
		netpollupdate(pd, _POLLIN, 0)
	case 'w':
		netpollupdate(pd, _POLLOUT, 0)
	default:
		throw("runtime: bad mode")
	}
	unlock(&pd.lock)
}

// netpollBreak interrupts a port_getn wait.
func netpollBreak() {
	// Failing to cas indicates there is an in-flight wakeup, so we're done here.
	if !netpollWakeSig.CompareAndSwap(0, 1) {
		return
	}

	// Use port_alert to put portfd into alert mode.
	// This will wake up all threads sleeping in port_getn on portfd,
	// and cause their calls to port_getn to return immediately.
	// Further, until portfd is taken out of alert mode,
	// all calls to port_getn will return immediately.
	if port_alert(portfd, _PORT_ALERT_UPDATE, _POLLHUP, uintptr(unsafe.Pointer(&portfd))) < 0 {
		if e := errno(); e != _EBUSY {
			println("runtime: port_alert failed with", e)
			throw("runtime: netpoll: port_alert failed")
		}
	}
}

// netpoll checks for ready network connections.
// Returns list of goroutines that become runnable.
// delay < 0: blocks indefinitely
// delay == 0: does not block, just polls
// delay > 0: block for up to that many nanoseconds
func netpoll(delay int64) (gList, int32) {
	if portfd == -1 {
		return gList{}, 0
	}

	var wait *timespec
	var ts timespec
	if delay < 0 {
		wait = nil
	} else if delay == 0 {
		wait = &ts
	} else {
		ts.setNsec(delay)
		if ts.tv_sec > 1e6 {
			// An arbitrary cap on how long to wait for a timer.
			// 1e6 s == ~11.5 days.
			ts.tv_sec = 1e6
		}
		wait = &ts
	}

	var events [128]portevent
retry:
	var n uint32 = 1
	r := port_getn(portfd, &events[0], uint32(len(events)), &n, wait)
	e := errno()
	if r < 0 && e == _ETIME && n > 0 {
		// As per port_getn(3C), an ETIME failure does not preclude the
		// delivery of some number of events.  Treat a timeout failure
		// with delivered events as a success.
		r = 0
	}
	if r < 0 {
		if e != _EINTR && e != _ETIME {
			print("runtime: port_getn on fd ", portfd, " failed (errno=", e, ")\n")
			throw("runtime: netpoll failed")
		}
		// If a timed sleep was interrupted and there are no events,
		// just return to recalculate how long we should sleep now.
		if delay > 0 {
			return gList{}, 0
		}
		goto retry
	}

	var toRun gList
	delta := int32(0)
	for i := 0; i < int(n); i++ {
		ev := &events[i]

		if ev.portev_source == _PORT_SOURCE_ALERT {
			if ev.portev_events != _POLLHUP || unsafe.Pointer(ev.portev_user) != unsafe.Pointer(&portfd) {
				throw("runtime: netpoll: bad port_alert wakeup")
			}
			if delay != 0 {
				// Now that a blocking call to netpoll
				// has seen the alert, take portfd
				// back out of alert mode.
				// See the comment in netpollBreak.
				if port_alert(portfd, 0, 0, 0) < 0 {
					e := errno()
					println("runtime: port_alert failed with", e)
					throw("runtime: netpoll: port_alert failed")
				}
				netpollWakeSig.Store(0)
			}
			continue
		}

		if ev.portev_events == 0 {
			continue
		}

		tp := taggedPointer(uintptr(unsafe.Pointer(ev.portev_user)))
		pd := (*pollDesc)(tp.pointer())
		if pd.fdseq.Load() != tp.tag() {
			continue
		}

		var mode, clear int32
		if (ev.portev_events & (_POLLIN | _POLLHUP | _POLLERR)) != 0 {
			mode += 'r'
			clear |= _POLLIN
		}
		if (ev.portev_events & (_POLLOUT | _POLLHUP | _POLLERR)) != 0 {
			mode += 'w'
			clear |= _POLLOUT
		}
		// To effect edge-triggered events, we need to be sure to
		// update our association with whatever events were not
		// set with the event. For example if we are registered
		// for POLLIN|POLLOUT, and we get POLLIN, besides waking
		// the goroutine interested in POLLIN we have to not forget
		// about the one interested in POLLOUT.
		if clear != 0 {
			lock(&pd.lock)
			netpollupdate(pd, 0, uint32(clear))
			unlock(&pd.lock)
		}

		if mode != 0 {
			// TODO(mikio): Consider implementing event
			// scanning error reporting once we are sure
			// about the event port on SmartOS.
			//
			// See golang.org/x/issue/30840.
			delta += netpollready(&toRun, pd, mode)
		}
	}

	return toRun, delta
}

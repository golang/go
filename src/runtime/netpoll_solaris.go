// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

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

//go:linkname libc_port_create libc_port_create
//go:linkname libc_port_associate libc_port_associate
//go:linkname libc_port_dissociate libc_port_dissociate
//go:linkname libc_port_getn libc_port_getn

var (
	libc_port_create,
	libc_port_associate,
	libc_port_dissociate,
	libc_port_getn libcFunc
)

func errno() int32 {
	return *getg().m.perrno
}

func fcntl(fd, cmd int32, arg uintptr) int32 {
	return int32(sysvicall3(&libc_fcntl, uintptr(fd), uintptr(cmd), arg))
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

var portfd int32 = -1

func netpollinit() {
	portfd = port_create()
	if portfd >= 0 {
		fcntl(portfd, _F_SETFD, _FD_CLOEXEC)
		return
	}

	print("netpollinit: failed to create port (", errno(), ")\n")
	throw("netpollinit: failed to create port")
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
	r := port_associate(portfd, _PORT_SOURCE_FD, fd, 0, uintptr(unsafe.Pointer(pd)))
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
	if pd.closing {
		return
	}

	old := pd.user
	events := (old & ^clear) | set
	if old == events {
		return
	}

	if events != 0 && port_associate(portfd, _PORT_SOURCE_FD, pd.fd, events, uintptr(unsafe.Pointer(pd))) != 0 {
		print("netpollupdate: failed to associate (", errno(), ")\n")
		throw("netpollupdate: failed to associate")
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
		throw("netpollarm: bad mode")
	}
	unlock(&pd.lock)
}

// polls for ready network connections
// returns list of goroutines that become runnable
func netpoll(block bool) *g {
	if portfd == -1 {
		return nil
	}

	var wait *timespec
	var zero timespec
	if !block {
		wait = &zero
	}

	var events [128]portevent
retry:
	var n uint32 = 1
	if port_getn(portfd, &events[0], uint32(len(events)), &n, wait) < 0 {
		if e := errno(); e != _EINTR {
			print("runtime: port_getn on fd ", portfd, " failed with ", e, "\n")
			throw("port_getn failed")
		}
		goto retry
	}

	var gp guintptr
	for i := 0; i < int(n); i++ {
		ev := &events[i]

		if ev.portev_events == 0 {
			continue
		}
		pd := (*pollDesc)(unsafe.Pointer(ev.portev_user))

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
			netpollready(&gp, pd, mode)
		}
	}

	if block && gp == 0 {
		goto retry
	}
	return gp.ptr()
}

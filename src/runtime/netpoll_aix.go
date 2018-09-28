// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

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

//go:nosplit
func fcntl(fd, cmd int32, arg uintptr) int32 {
	r, _ := syscall3(&libc_fcntl, uintptr(fd), uintptr(cmd), arg)
	return int32(r)
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
const _O_NONBLOCK = 0x4

var (
	pfds           []pollfd
	pds            []*pollDesc
	mtxpoll        mutex
	mtxset         mutex
	rdwake         int32
	wrwake         int32
	pendingUpdates int32
)

const pollVerbose = false

func netpollinit() {
	var p [2]int32

	// Create the pipe we use to wakeup poll.
	if err := pipe(&p[0]); err < 0 {
		throw("netpollinit: failed to create pipe")
	}
	rdwake = p[0]
	wrwake = p[1]

	fl := uintptr(fcntl(rdwake, _F_GETFL, 0))
	fcntl(rdwake, _F_SETFL, fl|_O_NONBLOCK)
	fcntl(rdwake, _F_SETFD, _FD_CLOEXEC)

	fl = uintptr(fcntl(wrwake, _F_GETFL, 0))
	fcntl(wrwake, _F_SETFL, fl|_O_NONBLOCK)
	fcntl(wrwake, _F_SETFD, _FD_CLOEXEC)

	// Pre-allocate array of pollfd structures for poll.
	if pollVerbose {
		println("*** allocating")
	}
	pfds = make([]pollfd, 1, 128)
	if pollVerbose {
		println("*** allocating done", &pfds[0])
	}

	// Poll the read side of the pipe.
	pfds[0].fd = rdwake
	pfds[0].events = _POLLIN

	pds = make([]*pollDesc, 1, 128)
	pds[0] = nil
}

func netpolldescriptor() uintptr {
	// Both fd must be returned
	if rdwake > 0xFFFF || wrwake > 0xFFFF {
		throw("netpolldescriptor: invalid fd number")
	}
	return uintptr(rdwake<<16 | wrwake)
}

// netpollwakeup writes on wrwake to wakeup poll before any changes.
func netpollwakeup() {
	if pendingUpdates == 0 {
		pendingUpdates = 1
		if pollVerbose {
			println("*** writing 1 byte")
		}
		b := [1]byte{0}
		write(uintptr(wrwake), unsafe.Pointer(&b[0]), 1)
	}
}

func netpollopen(fd uintptr, pd *pollDesc) int32 {
	if pollVerbose {
		println("*** netpollopen", fd)
	}
	lock(&mtxpoll)
	netpollwakeup()

	lock(&mtxset)
	unlock(&mtxpoll)

	pd.user = uint32(len(pfds))
	pfds = append(pfds, pollfd{fd: int32(fd)})
	pds = append(pds, pd)
	unlock(&mtxset)
	return 0
}

func netpollclose(fd uintptr) int32 {
	if pollVerbose {
		println("*** netpollclose", fd)
	}
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
	if pollVerbose {
		println("*** netpollarm", pd.fd, mode)
	}
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

//go:nowritebarrierrec
func netpoll(block bool) gList {
	timeout := ^uintptr(0)
	if !block {
		timeout = 0
		return gList{}
	}
	if pollVerbose {
		println("*** netpoll", block)
	}
retry:
	lock(&mtxpoll)
	lock(&mtxset)
	pendingUpdates = 0
	unlock(&mtxpoll)

	if pollVerbose {
		println("*** netpoll before poll")
	}
	n, e := poll(&pfds[0], uintptr(len(pfds)), timeout)
	if pollVerbose {
		println("*** netpoll after poll", n)
	}
	if n < 0 {
		if e != _EINTR {
			println("errno=", e, " len(pfds)=", len(pfds))
			throw("poll failed")
		}
		if pollVerbose {
			println("*** poll failed")
		}
		unlock(&mtxset)
		goto retry
	}
	// Check if some descriptors need to be changed
	if n != 0 && pfds[0].revents&(_POLLIN|_POLLHUP|_POLLERR) != 0 {
		var b [1]byte
		for read(rdwake, unsafe.Pointer(&b[0]), 1) == 1 {
			if pollVerbose {
				println("*** read 1 byte from pipe")
			}
		}
		// Do not look at the other fds in this case as the mode may have changed
		// XXX only additions of flags are made, so maybe it is ok
		unlock(&mtxset)
		goto retry
	}
	var toRun gList
	for i := 0; i < len(pfds) && n > 0; i++ {
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
			if pollVerbose {
				println("*** netpollready i=", i, "revents=", pfd.revents, "events=", pfd.events, "pd=", pds[i])
			}
			netpollready(&toRun, pds[i], mode)
			n--
		}
	}
	unlock(&mtxset)
	if block && toRun.empty() {
		goto retry
	}
	if pollVerbose {
		println("*** netpoll returning end")
	}
	return toRun
}

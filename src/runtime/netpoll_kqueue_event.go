// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd

package runtime

// Magic number of identifier used for EVFILT_USER.
// This number had zero Google results when it's created.
// That way, people will be directed here when this number
// get printed somehow and they search for it.
const kqIdent = 0xee1eb9f4

func addWakeupEvent(kq int32) {
	ev := keventt{
		ident:  kqIdent,
		filter: _EVFILT_USER,
		flags:  _EV_ADD | _EV_CLEAR,
	}
	for {
		n := kevent(kq, &ev, 1, nil, 0, nil)
		if n == 0 {
			break
		}
		if n == -_EINTR {
			// All changes contained in the changelist should have been applied
			// before returning EINTR. But let's be skeptical and retry it anyway,
			// to make a 100% commitment.
			continue
		}
		println("runtime: kevent for EVFILT_USER failed with", -n)
		throw("runtime: kevent failed")
	}
}

func wakeNetpoll(kq int32) {
	ev := keventt{
		ident:  kqIdent,
		filter: _EVFILT_USER,
		fflags: _NOTE_TRIGGER,
	}
	for {
		n := kevent(kq, &ev, 1, nil, 0, nil)
		if n == 0 {
			break
		}
		if n == -_EINTR {
			// Check out the comment in addWakeupEvent.
			continue
		}
		println("runtime: netpollBreak write failed with", -n)
		throw("runtime: netpollBreak write failed")
	}
}

func isWakeup(ev *keventt) bool {
	if ev.filter == _EVFILT_USER {
		if ev.ident == kqIdent {
			return true
		}
		println("runtime: netpoll: break fd ready for", ev.ident)
		throw("runtime: netpoll: break fd ready for something unexpected")
	}
	return false
}

func processWakeupEvent(kq int32, isBlocking bool) {
	if !isBlocking {
		// Got a wrong thread, relay
		wakeNetpoll(kq)
	}
}

func netpollIsPollDescriptor(fd uintptr) bool {
	return fd == uintptr(kq)
}

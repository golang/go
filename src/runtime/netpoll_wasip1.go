// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasip1

package runtime

import "unsafe"

// WASI network poller.
//
// WASI preview 1 includes a poll_oneoff host function that behaves similarly
// to poll(2) on Linux. Like poll(2), poll_oneoff is level triggered. It
// accepts one or more subscriptions to FD read or write events.
//
// Major differences to poll(2):
// - the events are not written to the input entries (like pollfd.revents), and
//   instead are appended to a separate events buffer. poll_oneoff writes zero
//   or more events to the buffer (at most one per input subscription) and
//   returns the number of events written. Although the index of the
//   subscriptions might not match the index of the associated event in the
//   events buffer, both the subscription and event structs contain a userdata
//   field and when a subscription yields an event the userdata fields will
//   match.
// - there's no explicit timeout parameter, although a time limit can be added
//   by using "clock" subscriptions.
// - each FD subscription can either be for a read or a write, but not both.
//   This is in contrast to poll(2) which accepts a mask with POLLIN and
//   POLLOUT bits, allowing for a subscription to either, neither, or both
//   reads and writes.
//
// Since poll_oneoff is similar to poll(2), the implementation here was derived
// from netpoll_aix.go.

const _EINTR = 27

var (
	evts []event
	subs []subscription
	pds  []*pollDesc
	mtx  mutex
)

func netpollinit() {
	// Unlike poll(2), WASI's poll_oneoff doesn't accept a timeout directly. To
	// prevent it from blocking indefinitely, a clock subscription with a
	// timeout field needs to be submitted. Reserve a slot here for the clock
	// subscription, and set fields that won't change between poll_oneoff calls.

	subs = make([]subscription, 1, 128)
	evts = make([]event, 0, 128)
	pds = make([]*pollDesc, 0, 128)

	timeout := &subs[0]
	eventtype := timeout.u.eventtype()
	*eventtype = eventtypeClock
	clock := timeout.u.subscriptionClock()
	clock.id = clockMonotonic
	clock.precision = 1e3
}

func netpollIsPollDescriptor(fd uintptr) bool {
	return false
}

func netpollopen(fd uintptr, pd *pollDesc) int32 {
	lock(&mtx)

	// We don't worry about pd.fdseq here,
	// as mtx protects us from stale pollDescs.

	pds = append(pds, pd)

	// The 32-bit pd.user field holds the index of the read subscription in the
	// upper 16 bits, and index of the write subscription in the lower bits.
	// A disarmed=^uint16(0) sentinel is used to represent no subscription.
	// There is thus a maximum of 65535 total subscriptions.
	pd.user = uint32(disarmed)<<16 | uint32(disarmed)

	unlock(&mtx)
	return 0
}

const disarmed = 0xFFFF

func netpollarm(pd *pollDesc, mode int) {
	lock(&mtx)

	var s subscription

	s.userdata = userdata(uintptr(unsafe.Pointer(pd)))

	fdReadwrite := s.u.subscriptionFdReadwrite()
	fdReadwrite.fd = int32(pd.fd)

	ridx := int(pd.user >> 16)
	widx := int(pd.user & 0xFFFF)

	if (mode == 'r' && ridx != disarmed) || (mode == 'w' && widx != disarmed) {
		unlock(&mtx)
		return
	}

	eventtype := s.u.eventtype()
	switch mode {
	case 'r':
		*eventtype = eventtypeFdRead
		ridx = len(subs)
	case 'w':
		*eventtype = eventtypeFdWrite
		widx = len(subs)
	}

	if len(subs) == disarmed {
		throw("overflow")
	}

	pd.user = uint32(ridx)<<16 | uint32(widx)

	subs = append(subs, s)
	evts = append(evts, event{})

	unlock(&mtx)
}

func netpolldisarm(pd *pollDesc, mode int32) {
	switch mode {
	case 'r':
		removesub(int(pd.user >> 16))
	case 'w':
		removesub(int(pd.user & 0xFFFF))
	case 'r' + 'w':
		removesub(int(pd.user >> 16))
		removesub(int(pd.user & 0xFFFF))
	}
}

func removesub(i int) {
	if i == disarmed {
		return
	}
	j := len(subs) - 1

	pdi := (*pollDesc)(unsafe.Pointer(uintptr(subs[i].userdata)))
	pdj := (*pollDesc)(unsafe.Pointer(uintptr(subs[j].userdata)))

	swapsub(pdi, i, disarmed)
	swapsub(pdj, j, i)

	subs = subs[:j]
}

func swapsub(pd *pollDesc, from, to int) {
	if from == to {
		return
	}
	ridx := int(pd.user >> 16)
	widx := int(pd.user & 0xFFFF)
	if ridx == from {
		ridx = to
	} else if widx == from {
		widx = to
	}
	pd.user = uint32(ridx)<<16 | uint32(widx)
	if to != disarmed {
		subs[to], subs[from] = subs[from], subs[to]
	}
}

func netpollclose(fd uintptr) int32 {
	lock(&mtx)
	for i := 0; i < len(pds); i++ {
		if pds[i].fd == fd {
			netpolldisarm(pds[i], 'r'+'w')
			pds[i] = pds[len(pds)-1]
			pds = pds[:len(pds)-1]
			break
		}
	}
	unlock(&mtx)
	return 0
}

func netpollBreak() {}

func netpoll(delay int64) (gList, int32) {
	lock(&mtx)

	// If delay >= 0, we include a subscription of type Clock that we use as
	// a timeout. If delay < 0, we omit the subscription and allow poll_oneoff
	// to block indefinitely.
	pollsubs := subs
	if delay >= 0 {
		timeout := &subs[0]
		clock := timeout.u.subscriptionClock()
		clock.timeout = uint64(delay)
	} else {
		pollsubs = subs[1:]
	}

	if len(pollsubs) == 0 {
		unlock(&mtx)
		return gList{}, 0
	}

	evts = evts[:len(pollsubs)]
	clear(evts)

retry:
	var nevents size
	errno := poll_oneoff(&pollsubs[0], &evts[0], uint32(len(pollsubs)), &nevents)
	if errno != 0 {
		if errno != _EINTR {
			println("errno=", errno, " len(pollsubs)=", len(pollsubs))
			throw("poll_oneoff failed")
		}
		// If a timed sleep was interrupted, just return to
		// recalculate how long we should sleep now.
		if delay > 0 {
			unlock(&mtx)
			return gList{}, 0
		}
		goto retry
	}

	var toRun gList
	delta := int32(0)
	for i := 0; i < int(nevents); i++ {
		e := &evts[i]
		if e.typ == eventtypeClock {
			continue
		}

		hangup := e.fdReadwrite.flags&fdReadwriteHangup != 0
		var mode int32
		if e.typ == eventtypeFdRead || e.error != 0 || hangup {
			mode += 'r'
		}
		if e.typ == eventtypeFdWrite || e.error != 0 || hangup {
			mode += 'w'
		}
		if mode != 0 {
			pd := (*pollDesc)(unsafe.Pointer(uintptr(e.userdata)))
			netpolldisarm(pd, mode)
			pd.setEventErr(e.error != 0, 0)
			delta += netpollready(&toRun, pd, mode)
		}
	}

	unlock(&mtx)
	return toRun, delta
}

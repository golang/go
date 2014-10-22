// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux nacl netbsd openbsd solaris windows

package runtime

import "unsafe"

// Integrated network poller (platform-independent part).
// A particular implementation (epoll/kqueue) must define the following functions:
// func netpollinit()			// to initialize the poller
// func netpollopen(fd uintptr, pd *pollDesc) int32	// to arm edge-triggered notifications
// and associate fd with pd.
// An implementation must call the following function to denote that the pd is ready.
// func netpollready(gpp **g, pd *pollDesc, mode int32)

// pollDesc contains 2 binary semaphores, rg and wg, to park reader and writer
// goroutines respectively. The semaphore can be in the following states:
// pdReady - io readiness notification is pending;
//           a goroutine consumes the notification by changing the state to nil.
// pdWait - a goroutine prepares to park on the semaphore, but not yet parked;
//          the goroutine commits to park by changing the state to G pointer,
//          or, alternatively, concurrent io notification changes the state to READY,
//          or, alternatively, concurrent timeout/close changes the state to nil.
// G pointer - the goroutine is blocked on the semaphore;
//             io notification or timeout/close changes the state to READY or nil respectively
//             and unparks the goroutine.
// nil - nothing of the above.
const (
	pdReady uintptr = 1
	pdWait  uintptr = 2
)

const pollBlockSize = 4 * 1024

// Network poller descriptor.
type pollDesc struct {
	link *pollDesc // in pollcache, protected by pollcache.lock

	// The lock protects pollOpen, pollSetDeadline, pollUnblock and deadlineimpl operations.
	// This fully covers seq, rt and wt variables. fd is constant throughout the PollDesc lifetime.
	// pollReset, pollWait, pollWaitCanceled and runtime·netpollready (IO readiness notification)
	// proceed w/o taking the lock. So closing, rg, rd, wg and wd are manipulated
	// in a lock-free way by all operations.
	// NOTE(dvyukov): the following code uses uintptr to store *g (rg/wg),
	// that will blow up when GC starts moving objects.
	lock    mutex // protectes the following fields
	fd      uintptr
	closing bool
	seq     uintptr        // protects from stale timers and ready notifications
	rg      uintptr        // pdReady, pdWait, G waiting for read or nil
	rt      timer          // read deadline timer (set if rt.f != nil)
	rd      int64          // read deadline
	wg      uintptr        // pdReady, pdWait, G waiting for write or nil
	wt      timer          // write deadline timer
	wd      int64          // write deadline
	user    unsafe.Pointer // user settable cookie
}

type pollCache struct {
	lock  mutex
	first *pollDesc
	// PollDesc objects must be type-stable,
	// because we can get ready notification from epoll/kqueue
	// after the descriptor is closed/reused.
	// Stale notifications are detected using seq variable,
	// seq is incremented when deadlines are changed or descriptor is reused.
}

var pollcache pollCache

func netpollServerInit() {
	onM(netpollinit)
}

func netpollOpen(fd uintptr) (*pollDesc, int) {
	pd := pollcache.alloc()
	lock(&pd.lock)
	if pd.wg != 0 && pd.wg != pdReady {
		gothrow("netpollOpen: blocked write on free descriptor")
	}
	if pd.rg != 0 && pd.rg != pdReady {
		gothrow("netpollOpen: blocked read on free descriptor")
	}
	pd.fd = fd
	pd.closing = false
	pd.seq++
	pd.rg = 0
	pd.rd = 0
	pd.wg = 0
	pd.wd = 0
	unlock(&pd.lock)

	var errno int32
	onM(func() {
		errno = netpollopen(fd, pd)
	})
	return pd, int(errno)
}

func netpollClose(pd *pollDesc) {
	if !pd.closing {
		gothrow("netpollClose: close w/o unblock")
	}
	if pd.wg != 0 && pd.wg != pdReady {
		gothrow("netpollClose: blocked write on closing descriptor")
	}
	if pd.rg != 0 && pd.rg != pdReady {
		gothrow("netpollClose: blocked read on closing descriptor")
	}
	onM(func() {
		netpollclose(uintptr(pd.fd))
	})
	pollcache.free(pd)
}

func (c *pollCache) free(pd *pollDesc) {
	lock(&c.lock)
	pd.link = c.first
	c.first = pd
	unlock(&c.lock)
}

func netpollReset(pd *pollDesc, mode int) int {
	err := netpollcheckerr(pd, int32(mode))
	if err != 0 {
		return err
	}
	if mode == 'r' {
		pd.rg = 0
	} else if mode == 'w' {
		pd.wg = 0
	}
	return 0
}

func netpollWait(pd *pollDesc, mode int) int {
	err := netpollcheckerr(pd, int32(mode))
	if err != 0 {
		return err
	}
	// As for now only Solaris uses level-triggered IO.
	if GOOS == "solaris" {
		onM(func() {
			netpollarm(pd, mode)
		})
	}
	for !netpollblock(pd, int32(mode), false) {
		err = netpollcheckerr(pd, int32(mode))
		if err != 0 {
			return err
		}
		// Can happen if timeout has fired and unblocked us,
		// but before we had a chance to run, timeout has been reset.
		// Pretend it has not happened and retry.
	}
	return 0
}

func netpollWaitCanceled(pd *pollDesc, mode int) {
	// This function is used only on windows after a failed attempt to cancel
	// a pending async IO operation. Wait for ioready, ignore closing or timeouts.
	for !netpollblock(pd, int32(mode), true) {
	}
}

func netpollSetDeadline(pd *pollDesc, d int64, mode int) {
	lock(&pd.lock)
	if pd.closing {
		unlock(&pd.lock)
		return
	}
	pd.seq++ // invalidate current timers
	// Reset current timers.
	if pd.rt.f != nil {
		deltimer(&pd.rt)
		pd.rt.f = nil
	}
	if pd.wt.f != nil {
		deltimer(&pd.wt)
		pd.wt.f = nil
	}
	// Setup new timers.
	if d != 0 && d <= nanotime() {
		d = -1
	}
	if mode == 'r' || mode == 'r'+'w' {
		pd.rd = d
	}
	if mode == 'w' || mode == 'r'+'w' {
		pd.wd = d
	}
	if pd.rd > 0 && pd.rd == pd.wd {
		pd.rt.f = netpollDeadline
		pd.rt.when = pd.rd
		// Copy current seq into the timer arg.
		// Timer func will check the seq against current descriptor seq,
		// if they differ the descriptor was reused or timers were reset.
		pd.rt.arg = pd
		pd.rt.seq = pd.seq
		addtimer(&pd.rt)
	} else {
		if pd.rd > 0 {
			pd.rt.f = netpollReadDeadline
			pd.rt.when = pd.rd
			pd.rt.arg = pd
			pd.rt.seq = pd.seq
			addtimer(&pd.rt)
		}
		if pd.wd > 0 {
			pd.wt.f = netpollWriteDeadline
			pd.wt.when = pd.wd
			pd.wt.arg = pd
			pd.wt.seq = pd.seq
			addtimer(&pd.wt)
		}
	}
	// If we set the new deadline in the past, unblock currently pending IO if any.
	var rg, wg *g
	atomicstorep(unsafe.Pointer(&wg), nil) // full memory barrier between stores to rd/wd and load of rg/wg in netpollunblock
	if pd.rd < 0 {
		rg = netpollunblock(pd, 'r', false)
	}
	if pd.wd < 0 {
		wg = netpollunblock(pd, 'w', false)
	}
	unlock(&pd.lock)
	if rg != nil {
		goready(rg)
	}
	if wg != nil {
		goready(wg)
	}
}

func netpollUnblock(pd *pollDesc) {
	lock(&pd.lock)
	if pd.closing {
		gothrow("netpollUnblock: already closing")
	}
	pd.closing = true
	pd.seq++
	var rg, wg *g
	atomicstorep(unsafe.Pointer(&rg), nil) // full memory barrier between store to closing and read of rg/wg in netpollunblock
	rg = netpollunblock(pd, 'r', false)
	wg = netpollunblock(pd, 'w', false)
	if pd.rt.f != nil {
		deltimer(&pd.rt)
		pd.rt.f = nil
	}
	if pd.wt.f != nil {
		deltimer(&pd.wt)
		pd.wt.f = nil
	}
	unlock(&pd.lock)
	if rg != nil {
		goready(rg)
	}
	if wg != nil {
		goready(wg)
	}
}

func netpollfd(pd *pollDesc) uintptr {
	return pd.fd
}

func netpolluser(pd *pollDesc) *unsafe.Pointer {
	return &pd.user
}

func netpollclosing(pd *pollDesc) bool {
	return pd.closing
}

func netpolllock(pd *pollDesc) {
	lock(&pd.lock)
}

func netpollunlock(pd *pollDesc) {
	unlock(&pd.lock)
}

// make pd ready, newly runnable goroutines (if any) are returned in rg/wg
func netpollready(gpp **g, pd *pollDesc, mode int32) {
	var rg, wg *g
	if mode == 'r' || mode == 'r'+'w' {
		rg = netpollunblock(pd, 'r', true)
	}
	if mode == 'w' || mode == 'r'+'w' {
		wg = netpollunblock(pd, 'w', true)
	}
	if rg != nil {
		rg.schedlink = *gpp
		*gpp = rg
	}
	if wg != nil {
		wg.schedlink = *gpp
		*gpp = wg
	}
}

func netpollcheckerr(pd *pollDesc, mode int32) int {
	if pd.closing {
		return 1 // errClosing
	}
	if (mode == 'r' && pd.rd < 0) || (mode == 'w' && pd.wd < 0) {
		return 2 // errTimeout
	}
	return 0
}

func netpollblockcommit(gp *g, gpp unsafe.Pointer) bool {
	return casuintptr((*uintptr)(gpp), pdWait, uintptr(unsafe.Pointer(gp)))
}

// returns true if IO is ready, or false if timedout or closed
// waitio - wait only for completed IO, ignore errors
func netpollblock(pd *pollDesc, mode int32, waitio bool) bool {
	gpp := &pd.rg
	if mode == 'w' {
		gpp = &pd.wg
	}

	// set the gpp semaphore to WAIT
	for {
		old := *gpp
		if old == pdReady {
			*gpp = 0
			return true
		}
		if old != 0 {
			gothrow("netpollblock: double wait")
		}
		if casuintptr(gpp, 0, pdWait) {
			break
		}
	}

	// need to recheck error states after setting gpp to WAIT
	// this is necessary because runtime_pollUnblock/runtime_pollSetDeadline/deadlineimpl
	// do the opposite: store to closing/rd/wd, membarrier, load of rg/wg
	if waitio || netpollcheckerr(pd, mode) == 0 {
		f := netpollblockcommit
		gopark(**(**unsafe.Pointer)(unsafe.Pointer(&f)), unsafe.Pointer(gpp), "IO wait")
	}
	// be careful to not lose concurrent READY notification
	old := xchguintptr(gpp, 0)
	if old > pdWait {
		gothrow("netpollblock: corrupted state")
	}
	return old == pdReady
}

func netpollunblock(pd *pollDesc, mode int32, ioready bool) *g {
	gpp := &pd.rg
	if mode == 'w' {
		gpp = &pd.wg
	}

	for {
		old := *gpp
		if old == pdReady {
			return nil
		}
		if old == 0 && !ioready {
			// Only set READY for ioready. runtime_pollWait
			// will check for timeout/cancel before waiting.
			return nil
		}
		var new uintptr
		if ioready {
			new = pdReady
		}
		if casuintptr(gpp, old, new) {
			if old == pdReady || old == pdWait {
				old = 0
			}
			return (*g)(unsafe.Pointer(old))
		}
	}
}

func netpolldeadlineimpl(pd *pollDesc, seq uintptr, read, write bool) {
	lock(&pd.lock)
	// Seq arg is seq when the timer was set.
	// If it's stale, ignore the timer event.
	if seq != pd.seq {
		// The descriptor was reused or timers were reset.
		unlock(&pd.lock)
		return
	}
	var rg *g
	if read {
		if pd.rd <= 0 || pd.rt.f == nil {
			gothrow("netpolldeadlineimpl: inconsistent read deadline")
		}
		pd.rd = -1
		atomicstorep(unsafe.Pointer(&pd.rt.f), nil) // full memory barrier between store to rd and load of rg in netpollunblock
		rg = netpollunblock(pd, 'r', false)
	}
	var wg *g
	if write {
		if pd.wd <= 0 || pd.wt.f == nil && !read {
			gothrow("netpolldeadlineimpl: inconsistent write deadline")
		}
		pd.wd = -1
		atomicstorep(unsafe.Pointer(&pd.wt.f), nil) // full memory barrier between store to wd and load of wg in netpollunblock
		wg = netpollunblock(pd, 'w', false)
	}
	unlock(&pd.lock)
	if rg != nil {
		goready(rg)
	}
	if wg != nil {
		goready(wg)
	}
}

func netpollDeadline(arg interface{}, seq uintptr) {
	netpolldeadlineimpl(arg.(*pollDesc), seq, true, true)
}

func netpollReadDeadline(arg interface{}, seq uintptr) {
	netpolldeadlineimpl(arg.(*pollDesc), seq, true, false)
}

func netpollWriteDeadline(arg interface{}, seq uintptr) {
	netpolldeadlineimpl(arg.(*pollDesc), seq, false, true)
}

func (c *pollCache) alloc() *pollDesc {
	lock(&c.lock)
	if c.first == nil {
		const pdSize = unsafe.Sizeof(pollDesc{})
		n := pollBlockSize / pdSize
		if n == 0 {
			n = 1
		}
		// Must be in non-GC memory because can be referenced
		// only from epoll/kqueue internals.
		mem := persistentalloc(n*pdSize, 0, &memstats.other_sys)
		for i := uintptr(0); i < n; i++ {
			pd := (*pollDesc)(add(mem, i*pdSize))
			pd.link = c.first
			c.first = pd
		}
	}
	pd := c.first
	c.first = pd.link
	unlock(&c.lock)
	return pd
}

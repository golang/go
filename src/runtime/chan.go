// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// This file contains the implementation of Go channels.

import "unsafe"

const (
	maxAlign  = 8
	hchanSize = unsafe.Sizeof(hchan{}) + uintptr(-int(unsafe.Sizeof(hchan{}))&(maxAlign-1))
	debugChan = false
)

// TODO(khr): make hchan.buf an unsafe.Pointer, not a *uint8

//go:linkname reflect_makechan reflect.makechan
func reflect_makechan(t *chantype, size int64) *hchan {
	return makechan(t, size)
}

func makechan(t *chantype, size int64) *hchan {
	elem := t.elem

	// compiler checks this but be safe.
	if elem.size >= 1<<16 {
		gothrow("makechan: invalid channel element type")
	}
	if hchanSize%maxAlign != 0 || elem.align > maxAlign {
		gothrow("makechan: bad alignment")
	}
	if size < 0 || int64(uintptr(size)) != size || (elem.size > 0 && uintptr(size) > (_MaxMem-hchanSize)/uintptr(elem.size)) {
		panic("makechan: size out of range")
	}

	var c *hchan
	if elem.kind&kindNoPointers != 0 || size == 0 {
		// Allocate memory in one call.
		// Hchan does not contain pointers interesting for GC in this case:
		// buf points into the same allocation, elemtype is persistent.
		// SudoG's are referenced from their owning thread so they can't be collected.
		// TODO(dvyukov,rlh): Rethink when collector can move allocated objects.
		c = (*hchan)(mallocgc(hchanSize+uintptr(size)*uintptr(elem.size), nil, flagNoScan))
		if size > 0 && elem.size != 0 {
			c.buf = (*uint8)(add(unsafe.Pointer(c), hchanSize))
		} else {
			c.buf = (*uint8)(unsafe.Pointer(c)) // race detector uses this location for synchronization
		}
	} else {
		c = new(hchan)
		c.buf = (*uint8)(newarray(elem, uintptr(size)))
	}
	c.elemsize = uint16(elem.size)
	c.elemtype = elem
	c.dataqsiz = uint(size)

	if debugChan {
		print("makechan: chan=", c, "; elemsize=", elem.size, "; elemalg=", elem.alg, "; dataqsiz=", size, "\n")
	}
	return c
}

// chanbuf(c, i) is pointer to the i'th slot in the buffer.
func chanbuf(c *hchan, i uint) unsafe.Pointer {
	return add(unsafe.Pointer(c.buf), uintptr(i)*uintptr(c.elemsize))
}

// entry point for c <- x from compiled code
//go:nosplit
func chansend1(t *chantype, c *hchan, elem unsafe.Pointer) {
	chansend(t, c, elem, true, getcallerpc(unsafe.Pointer(&t)))
}

/*
 * generic single channel send/recv
 * If block is not nil,
 * then the protocol will not
 * sleep but return if it could
 * not complete.
 *
 * sleep can wake up with g.param == nil
 * when a channel involved in the sleep has
 * been closed.  it is easiest to loop and re-run
 * the operation; we'll see that it's now closed.
 */
func chansend(t *chantype, c *hchan, ep unsafe.Pointer, block bool, callerpc uintptr) bool {
	if raceenabled {
		raceReadObjectPC(t.elem, ep, callerpc, funcPC(chansend))
	}

	if c == nil {
		if !block {
			return false
		}
		gopark(nil, nil, "chan send (nil chan)")
		gothrow("unreachable")
	}

	if debugChan {
		print("chansend: chan=", c, "\n")
	}

	if raceenabled {
		racereadpc(unsafe.Pointer(c), callerpc, funcPC(chansend))
	}

	// Fast path: check for failed non-blocking operation without acquiring the lock.
	//
	// After observing that the channel is not closed, we observe that the channel is
	// not ready for sending. Each of these observations is a single word-sized read
	// (first c.closed and second c.recvq.first or c.qcount depending on kind of channel).
	// Because a closed channel cannot transition from 'ready for sending' to
	// 'not ready for sending', even if the channel is closed between the two observations,
	// they imply a moment between the two when the channel was both not yet closed
	// and not ready for sending. We behave as if we observed the channel at that moment,
	// and report that the send cannot proceed.
	//
	// It is okay if the reads are reordered here: if we observe that the channel is not
	// ready for sending and then observe that it is not closed, that implies that the
	// channel wasn't closed during the first observation.
	if !block && c.closed == 0 && ((c.dataqsiz == 0 && c.recvq.first == nil) ||
		(c.dataqsiz > 0 && c.qcount == c.dataqsiz)) {
		return false
	}

	var t0 int64
	if blockprofilerate > 0 {
		t0 = cputicks()
	}

	lock(&c.lock)
	if c.closed != 0 {
		unlock(&c.lock)
		panic("send on closed channel")
	}

	if c.dataqsiz == 0 { // synchronous channel
		sg := c.recvq.dequeue()
		if sg != nil { // found a waiting receiver
			if raceenabled {
				racesync(c, sg)
			}
			unlock(&c.lock)

			recvg := sg.g
			if sg.elem != nil {
				memmove(unsafe.Pointer(sg.elem), ep, uintptr(c.elemsize))
				sg.elem = nil
			}
			recvg.param = unsafe.Pointer(sg)
			if sg.releasetime != 0 {
				sg.releasetime = cputicks()
			}
			goready(recvg)
			return true
		}

		if !block {
			unlock(&c.lock)
			return false
		}

		// no receiver available: block on this channel.
		gp := getg()
		mysg := acquireSudog()
		mysg.releasetime = 0
		if t0 != 0 {
			mysg.releasetime = -1
		}
		mysg.elem = ep
		mysg.waitlink = nil
		gp.waiting = mysg
		mysg.g = gp
		mysg.selectdone = nil
		gp.param = nil
		c.sendq.enqueue(mysg)
		goparkunlock(&c.lock, "chan send")

		// someone woke us up.
		if mysg != gp.waiting {
			gothrow("G waiting list is corrupted!")
		}
		gp.waiting = nil
		if gp.param == nil {
			if c.closed == 0 {
				gothrow("chansend: spurious wakeup")
			}
			panic("send on closed channel")
		}
		gp.param = nil
		if mysg.releasetime > 0 {
			blockevent(int64(mysg.releasetime)-t0, 2)
		}
		releaseSudog(mysg)
		return true
	}

	// asynchronous channel
	// wait for some space to write our data
	var t1 int64
	for c.qcount >= c.dataqsiz {
		if !block {
			unlock(&c.lock)
			return false
		}
		gp := getg()
		mysg := acquireSudog()
		mysg.releasetime = 0
		if t0 != 0 {
			mysg.releasetime = -1
		}
		mysg.g = gp
		mysg.elem = nil
		mysg.selectdone = nil
		c.sendq.enqueue(mysg)
		goparkunlock(&c.lock, "chan send")

		// someone woke us up - try again
		if mysg.releasetime > 0 {
			t1 = mysg.releasetime
		}
		releaseSudog(mysg)
		lock(&c.lock)
		if c.closed != 0 {
			unlock(&c.lock)
			panic("send on closed channel")
		}
	}

	// write our data into the channel buffer
	if raceenabled {
		raceacquire(chanbuf(c, c.sendx))
		racerelease(chanbuf(c, c.sendx))
	}
	memmove(chanbuf(c, c.sendx), ep, uintptr(c.elemsize))
	c.sendx++
	if c.sendx == c.dataqsiz {
		c.sendx = 0
	}
	c.qcount++

	// wake up a waiting receiver
	sg := c.recvq.dequeue()
	if sg != nil {
		recvg := sg.g
		unlock(&c.lock)
		if sg.releasetime != 0 {
			sg.releasetime = cputicks()
		}
		goready(recvg)
	} else {
		unlock(&c.lock)
	}
	if t1 > 0 {
		blockevent(t1-t0, 2)
	}
	return true
}

func closechan(c *hchan) {
	if c == nil {
		panic("close of nil channel")
	}

	lock(&c.lock)
	if c.closed != 0 {
		unlock(&c.lock)
		panic("close of closed channel")
	}

	if raceenabled {
		callerpc := getcallerpc(unsafe.Pointer(&c))
		racewritepc(unsafe.Pointer(c), callerpc, funcPC(closechan))
		racerelease(unsafe.Pointer(c))
	}

	c.closed = 1

	// release all readers
	for {
		sg := c.recvq.dequeue()
		if sg == nil {
			break
		}
		gp := sg.g
		sg.elem = nil
		gp.param = nil
		if sg.releasetime != 0 {
			sg.releasetime = cputicks()
		}
		goready(gp)
	}

	// release all writers
	for {
		sg := c.sendq.dequeue()
		if sg == nil {
			break
		}
		gp := sg.g
		sg.elem = nil
		gp.param = nil
		if sg.releasetime != 0 {
			sg.releasetime = cputicks()
		}
		goready(gp)
	}
	unlock(&c.lock)
}

// entry points for <- c from compiled code
//go:nosplit
func chanrecv1(t *chantype, c *hchan, elem unsafe.Pointer) {
	chanrecv(t, c, elem, true)
}

//go:nosplit
func chanrecv2(t *chantype, c *hchan, elem unsafe.Pointer) (received bool) {
	_, received = chanrecv(t, c, elem, true)
	return
}

// chanrecv receives on channel c and writes the received data to ep.
// ep may be nil, in which case received data is ignored.
// If block == false and no elements are available, returns (false, false).
// Otherwise, if c is closed, zeros *ep and returns (true, false).
// Otherwise, fills in *ep with an element and returns (true, true).
func chanrecv(t *chantype, c *hchan, ep unsafe.Pointer, block bool) (selected, received bool) {
	// raceenabled: don't need to check ep, as it is always on the stack.

	if debugChan {
		print("chanrecv: chan=", c, "\n")
	}

	if c == nil {
		if !block {
			return
		}
		gopark(nil, nil, "chan receive (nil chan)")
		gothrow("unreachable")
	}

	// Fast path: check for failed non-blocking operation without acquiring the lock.
	//
	// After observing that the channel is not ready for receiving, we observe that the
	// channel is not closed. Each of these observations is a single word-sized read
	// (first c.sendq.first or c.qcount, and second c.closed).
	// Because a channel cannot be reopened, the later observation of the channel
	// being not closed implies that it was also not closed at the moment of the
	// first observation. We behave as if we observed the channel at that moment
	// and report that the receive cannot proceed.
	//
	// The order of operations is important here: reversing the operations can lead to
	// incorrect behavior when racing with a close.
	if !block && (c.dataqsiz == 0 && c.sendq.first == nil ||
		c.dataqsiz > 0 && atomicloaduint(&c.qcount) == 0) &&
		atomicload(&c.closed) == 0 {
		return
	}

	var t0 int64
	if blockprofilerate > 0 {
		t0 = cputicks()
	}

	lock(&c.lock)
	if c.dataqsiz == 0 { // synchronous channel
		if c.closed != 0 {
			return recvclosed(c, ep)
		}

		sg := c.sendq.dequeue()
		if sg != nil {
			if raceenabled {
				racesync(c, sg)
			}
			unlock(&c.lock)

			if ep != nil {
				memmove(ep, sg.elem, uintptr(c.elemsize))
			}
			sg.elem = nil
			gp := sg.g
			gp.param = unsafe.Pointer(sg)
			if sg.releasetime != 0 {
				sg.releasetime = cputicks()
			}
			goready(gp)
			selected = true
			received = true
			return
		}

		if !block {
			unlock(&c.lock)
			return
		}

		// no sender available: block on this channel.
		gp := getg()
		mysg := acquireSudog()
		mysg.releasetime = 0
		if t0 != 0 {
			mysg.releasetime = -1
		}
		mysg.elem = ep
		mysg.waitlink = nil
		gp.waiting = mysg
		mysg.g = gp
		mysg.selectdone = nil
		gp.param = nil
		c.recvq.enqueue(mysg)
		goparkunlock(&c.lock, "chan receive")

		// someone woke us up
		if mysg != gp.waiting {
			gothrow("G waiting list is corrupted!")
		}
		gp.waiting = nil
		if mysg.releasetime > 0 {
			blockevent(mysg.releasetime-t0, 2)
		}
		haveData := gp.param != nil
		gp.param = nil
		releaseSudog(mysg)

		if haveData {
			// a sender sent us some data. It already wrote to ep.
			selected = true
			received = true
			return
		}

		lock(&c.lock)
		if c.closed == 0 {
			gothrow("chanrecv: spurious wakeup")
		}
		return recvclosed(c, ep)
	}

	// asynchronous channel
	// wait for some data to appear
	var t1 int64
	for c.qcount <= 0 {
		if c.closed != 0 {
			selected, received = recvclosed(c, ep)
			if t1 > 0 {
				blockevent(t1-t0, 2)
			}
			return
		}

		if !block {
			unlock(&c.lock)
			return
		}

		// wait for someone to send an element
		gp := getg()
		mysg := acquireSudog()
		mysg.releasetime = 0
		if t0 != 0 {
			mysg.releasetime = -1
		}
		mysg.elem = nil
		mysg.g = gp
		mysg.selectdone = nil

		c.recvq.enqueue(mysg)
		goparkunlock(&c.lock, "chan receive")

		// someone woke us up - try again
		if mysg.releasetime > 0 {
			t1 = mysg.releasetime
		}
		releaseSudog(mysg)
		lock(&c.lock)
	}

	if raceenabled {
		raceacquire(chanbuf(c, c.recvx))
		racerelease(chanbuf(c, c.recvx))
	}
	if ep != nil {
		memmove(ep, chanbuf(c, c.recvx), uintptr(c.elemsize))
	}
	memclr(chanbuf(c, c.recvx), uintptr(c.elemsize))

	c.recvx++
	if c.recvx == c.dataqsiz {
		c.recvx = 0
	}
	c.qcount--

	// ping a sender now that there is space
	sg := c.sendq.dequeue()
	if sg != nil {
		gp := sg.g
		unlock(&c.lock)
		if sg.releasetime != 0 {
			sg.releasetime = cputicks()
		}
		goready(gp)
	} else {
		unlock(&c.lock)
	}

	if t1 > 0 {
		blockevent(t1-t0, 2)
	}
	selected = true
	received = true
	return
}

// recvclosed is a helper function for chanrecv.  Handles cleanup
// when the receiver encounters a closed channel.
// Caller must hold c.lock, recvclosed will release the lock.
func recvclosed(c *hchan, ep unsafe.Pointer) (selected, recevied bool) {
	if raceenabled {
		raceacquire(unsafe.Pointer(c))
	}
	unlock(&c.lock)
	if ep != nil {
		memclr(ep, uintptr(c.elemsize))
	}
	return true, false
}

// compiler implements
//
//	select {
//	case c <- v:
//		... foo
//	default:
//		... bar
//	}
//
// as
//
//	if selectnbsend(c, v) {
//		... foo
//	} else {
//		... bar
//	}
//
func selectnbsend(t *chantype, c *hchan, elem unsafe.Pointer) (selected bool) {
	return chansend(t, c, elem, false, getcallerpc(unsafe.Pointer(&t)))
}

// compiler implements
//
//	select {
//	case v = <-c:
//		... foo
//	default:
//		... bar
//	}
//
// as
//
//	if selectnbrecv(&v, c) {
//		... foo
//	} else {
//		... bar
//	}
//
func selectnbrecv(t *chantype, elem unsafe.Pointer, c *hchan) (selected bool) {
	selected, _ = chanrecv(t, c, elem, false)
	return
}

// compiler implements
//
//	select {
//	case v, ok = <-c:
//		... foo
//	default:
//		... bar
//	}
//
// as
//
//	if c != nil && selectnbrecv2(&v, &ok, c) {
//		... foo
//	} else {
//		... bar
//	}
//
func selectnbrecv2(t *chantype, elem unsafe.Pointer, received *bool, c *hchan) (selected bool) {
	// TODO(khr): just return 2 values from this function, now that it is in Go.
	selected, *received = chanrecv(t, c, elem, false)
	return
}

//go:linkname reflect_chansend reflect.chansend
func reflect_chansend(t *chantype, c *hchan, elem unsafe.Pointer, nb bool) (selected bool) {
	return chansend(t, c, elem, !nb, getcallerpc(unsafe.Pointer(&t)))
}

//go:linkname reflect_chanrecv reflect.chanrecv
func reflect_chanrecv(t *chantype, c *hchan, nb bool, elem unsafe.Pointer) (selected bool, received bool) {
	return chanrecv(t, c, elem, !nb)
}

//go:linkname reflect_chanlen reflect.chanlen
func reflect_chanlen(c *hchan) int {
	if c == nil {
		return 0
	}
	return int(c.qcount)
}

//go:linkname reflect_chancap reflect.chancap
func reflect_chancap(c *hchan) int {
	if c == nil {
		return 0
	}
	return int(c.dataqsiz)
}

//go:linkname reflect_chanclose reflect.chanclose
func reflect_chanclose(c *hchan) {
	closechan(c)
}

func (q *waitq) enqueue(sgp *sudog) {
	sgp.next = nil
	x := q.last
	if x == nil {
		sgp.prev = nil
		q.first = sgp
		q.last = sgp
		return
	}
	sgp.prev = x
	x.next = sgp
	q.last = sgp
}

func (q *waitq) dequeue() *sudog {
	for {
		sgp := q.first
		if sgp == nil {
			return nil
		}
		y := sgp.next
		if y == nil {
			q.first = nil
			q.last = nil
		} else {
			y.prev = nil
			q.first = y
			sgp.next = nil // mark as removed (see dequeueSudog)
		}

		// if sgp participates in a select and is already signaled, ignore it
		if sgp.selectdone != nil {
			// claim the right to signal
			if *sgp.selectdone != 0 || !cas(sgp.selectdone, 0, 1) {
				continue
			}
		}

		return sgp
	}
}

func racesync(c *hchan, sg *sudog) {
	racerelease(chanbuf(c, 0))
	raceacquireg(sg.g, chanbuf(c, 0))
	racereleaseg(sg.g, chanbuf(c, 0))
	raceacquire(chanbuf(c, 0))
}

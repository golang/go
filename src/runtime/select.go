// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// This file contains the implementation of Go select statements.

import "unsafe"

const (
	debugSelect = false
)

var (
	chansendpc = funcPC(chansend)
	chanrecvpc = funcPC(chanrecv)
)

func selectsize(size uintptr) uintptr {
	selsize := unsafe.Sizeof(_select{}) +
		(size-1)*unsafe.Sizeof(_select{}.scase[0]) +
		size*unsafe.Sizeof(*_select{}.lockorder) +
		size*unsafe.Sizeof(*_select{}.pollorder)
	return round(selsize, _Int64Align)
}

func newselect(sel *_select, selsize int64, size int32) {
	if selsize != int64(selectsize(uintptr(size))) {
		print("runtime: bad select size ", selsize, ", want ", selectsize(uintptr(size)), "\n")
		throw("bad select size")
	}
	sel.tcase = uint16(size)
	sel.ncase = 0
	sel.lockorder = (**hchan)(add(unsafe.Pointer(&sel.scase), uintptr(size)*unsafe.Sizeof(_select{}.scase[0])))
	sel.pollorder = (*uint16)(add(unsafe.Pointer(sel.lockorder), uintptr(size)*unsafe.Sizeof(*_select{}.lockorder)))

	if debugSelect {
		print("newselect s=", sel, " size=", size, "\n")
	}
}

//go:nosplit
func selectsend(sel *_select, c *hchan, elem unsafe.Pointer) (selected bool) {
	// nil cases do not compete
	if c != nil {
		selectsendImpl(sel, c, getcallerpc(unsafe.Pointer(&sel)), elem, uintptr(unsafe.Pointer(&selected))-uintptr(unsafe.Pointer(&sel)))
	}
	return
}

// cut in half to give stack a chance to split
func selectsendImpl(sel *_select, c *hchan, pc uintptr, elem unsafe.Pointer, so uintptr) {
	i := sel.ncase
	if i >= sel.tcase {
		throw("selectsend: too many cases")
	}
	sel.ncase = i + 1
	cas := (*scase)(add(unsafe.Pointer(&sel.scase), uintptr(i)*unsafe.Sizeof(sel.scase[0])))

	cas.pc = pc
	cas._chan = c
	cas.so = uint16(so)
	cas.kind = _CaseSend
	cas.elem = elem

	if debugSelect {
		print("selectsend s=", sel, " pc=", hex(cas.pc), " chan=", cas._chan, " so=", cas.so, "\n")
	}
}

//go:nosplit
func selectrecv(sel *_select, c *hchan, elem unsafe.Pointer) (selected bool) {
	// nil cases do not compete
	if c != nil {
		selectrecvImpl(sel, c, getcallerpc(unsafe.Pointer(&sel)), elem, nil, uintptr(unsafe.Pointer(&selected))-uintptr(unsafe.Pointer(&sel)))
	}
	return
}

//go:nosplit
func selectrecv2(sel *_select, c *hchan, elem unsafe.Pointer, received *bool) (selected bool) {
	// nil cases do not compete
	if c != nil {
		selectrecvImpl(sel, c, getcallerpc(unsafe.Pointer(&sel)), elem, received, uintptr(unsafe.Pointer(&selected))-uintptr(unsafe.Pointer(&sel)))
	}
	return
}

func selectrecvImpl(sel *_select, c *hchan, pc uintptr, elem unsafe.Pointer, received *bool, so uintptr) {
	i := sel.ncase
	if i >= sel.tcase {
		throw("selectrecv: too many cases")
	}
	sel.ncase = i + 1
	cas := (*scase)(add(unsafe.Pointer(&sel.scase), uintptr(i)*unsafe.Sizeof(sel.scase[0])))
	cas.pc = pc
	cas._chan = c
	cas.so = uint16(so)
	cas.kind = _CaseRecv
	cas.elem = elem
	cas.receivedp = received

	if debugSelect {
		print("selectrecv s=", sel, " pc=", hex(cas.pc), " chan=", cas._chan, " so=", cas.so, "\n")
	}
}

//go:nosplit
func selectdefault(sel *_select) (selected bool) {
	selectdefaultImpl(sel, getcallerpc(unsafe.Pointer(&sel)), uintptr(unsafe.Pointer(&selected))-uintptr(unsafe.Pointer(&sel)))
	return
}

func selectdefaultImpl(sel *_select, callerpc uintptr, so uintptr) {
	i := sel.ncase
	if i >= sel.tcase {
		throw("selectdefault: too many cases")
	}
	sel.ncase = i + 1
	cas := (*scase)(add(unsafe.Pointer(&sel.scase), uintptr(i)*unsafe.Sizeof(sel.scase[0])))
	cas.pc = callerpc
	cas._chan = nil
	cas.so = uint16(so)
	cas.kind = _CaseDefault

	if debugSelect {
		print("selectdefault s=", sel, " pc=", hex(cas.pc), " so=", cas.so, "\n")
	}
}

func sellock(sel *_select) {
	lockslice := sliceStruct{unsafe.Pointer(sel.lockorder), int(sel.ncase), int(sel.ncase)}
	lockorder := *(*[]*hchan)(unsafe.Pointer(&lockslice))
	var c *hchan
	for _, c0 := range lockorder {
		if c0 != nil && c0 != c {
			c = c0
			lock(&c.lock)
		}
	}
}

func selunlock(sel *_select) {
	// We must be very careful here to not touch sel after we have unlocked
	// the last lock, because sel can be freed right after the last unlock.
	// Consider the following situation.
	// First M calls runtime·park() in runtime·selectgo() passing the sel.
	// Once runtime·park() has unlocked the last lock, another M makes
	// the G that calls select runnable again and schedules it for execution.
	// When the G runs on another M, it locks all the locks and frees sel.
	// Now if the first M touches sel, it will access freed memory.
	n := int(sel.ncase)
	r := 0
	lockslice := sliceStruct{unsafe.Pointer(sel.lockorder), n, n}
	lockorder := *(*[]*hchan)(unsafe.Pointer(&lockslice))
	// skip the default case
	if n > 0 && lockorder[0] == nil {
		r = 1
	}
	for i := n - 1; i >= r; i-- {
		c := lockorder[i]
		if i > 0 && c == lockorder[i-1] {
			continue // will unlock it on the next iteration
		}
		unlock(&c.lock)
	}
}

func selparkcommit(gp *g, sel unsafe.Pointer) bool {
	selunlock((*_select)(sel))
	return true
}

func block() {
	gopark(nil, nil, "select (no cases)", traceEvGoStop) // forever
}

// overwrites return pc on stack to signal which case of the select
// to run, so cannot appear at the top of a split stack.
//go:nosplit
func selectgo(sel *_select) {
	pc, offset := selectgoImpl(sel)
	*(*bool)(add(unsafe.Pointer(&sel), uintptr(offset))) = true
	setcallerpc(unsafe.Pointer(&sel), pc)
}

// selectgoImpl returns scase.pc and scase.so for the select
// case which fired.
func selectgoImpl(sel *_select) (uintptr, uint16) {
	if debugSelect {
		print("select: sel=", sel, "\n")
	}

	scaseslice := sliceStruct{unsafe.Pointer(&sel.scase), int(sel.ncase), int(sel.ncase)}
	scases := *(*[]scase)(unsafe.Pointer(&scaseslice))

	var t0 int64
	if blockprofilerate > 0 {
		t0 = cputicks()
		for i := 0; i < int(sel.ncase); i++ {
			scases[i].releasetime = -1
		}
	}

	// The compiler rewrites selects that statically have
	// only 0 or 1 cases plus default into simpler constructs.
	// The only way we can end up with such small sel.ncase
	// values here is for a larger select in which most channels
	// have been nilled out.  The general code handles those
	// cases correctly, and they are rare enough not to bother
	// optimizing (and needing to test).

	// generate permuted order
	pollslice := sliceStruct{unsafe.Pointer(sel.pollorder), int(sel.ncase), int(sel.ncase)}
	pollorder := *(*[]uint16)(unsafe.Pointer(&pollslice))
	for i := 0; i < int(sel.ncase); i++ {
		pollorder[i] = uint16(i)
	}
	for i := 1; i < int(sel.ncase); i++ {
		o := pollorder[i]
		j := int(fastrand1()) % (i + 1)
		pollorder[i] = pollorder[j]
		pollorder[j] = o
	}

	// sort the cases by Hchan address to get the locking order.
	// simple heap sort, to guarantee n log n time and constant stack footprint.
	lockslice := sliceStruct{unsafe.Pointer(sel.lockorder), int(sel.ncase), int(sel.ncase)}
	lockorder := *(*[]*hchan)(unsafe.Pointer(&lockslice))
	for i := 0; i < int(sel.ncase); i++ {
		j := i
		c := scases[j]._chan
		for j > 0 && lockorder[(j-1)/2].sortkey() < c.sortkey() {
			k := (j - 1) / 2
			lockorder[j] = lockorder[k]
			j = k
		}
		lockorder[j] = c
	}
	for i := int(sel.ncase) - 1; i >= 0; i-- {
		c := lockorder[i]
		lockorder[i] = lockorder[0]
		j := 0
		for {
			k := j*2 + 1
			if k >= i {
				break
			}
			if k+1 < i && lockorder[k].sortkey() < lockorder[k+1].sortkey() {
				k++
			}
			if c.sortkey() < lockorder[k].sortkey() {
				lockorder[j] = lockorder[k]
				j = k
				continue
			}
			break
		}
		lockorder[j] = c
	}
	/*
		for i := 0; i+1 < int(sel.ncase); i++ {
			if lockorder[i].sortkey() > lockorder[i+1].sortkey() {
				print("i=", i, " x=", lockorder[i], " y=", lockorder[i+1], "\n")
				throw("select: broken sort")
			}
		}
	*/

	// lock all the channels involved in the select
	sellock(sel)

	var (
		gp     *g
		done   uint32
		sg     *sudog
		c      *hchan
		k      *scase
		sglist *sudog
		sgnext *sudog
	)

loop:
	// pass 1 - look for something already waiting
	var dfl *scase
	var cas *scase
	for i := 0; i < int(sel.ncase); i++ {
		cas = &scases[pollorder[i]]
		c = cas._chan

		switch cas.kind {
		case _CaseRecv:
			if c.dataqsiz > 0 {
				if c.qcount > 0 {
					goto asyncrecv
				}
			} else {
				sg = c.sendq.dequeue()
				if sg != nil {
					goto syncrecv
				}
			}
			if c.closed != 0 {
				goto rclose
			}

		case _CaseSend:
			if raceenabled {
				racereadpc(unsafe.Pointer(c), cas.pc, chansendpc)
			}
			if c.closed != 0 {
				goto sclose
			}
			if c.dataqsiz > 0 {
				if c.qcount < c.dataqsiz {
					goto asyncsend
				}
			} else {
				sg = c.recvq.dequeue()
				if sg != nil {
					goto syncsend
				}
			}

		case _CaseDefault:
			dfl = cas
		}
	}

	if dfl != nil {
		selunlock(sel)
		cas = dfl
		goto retc
	}

	// pass 2 - enqueue on all chans
	gp = getg()
	done = 0
	for i := 0; i < int(sel.ncase); i++ {
		cas = &scases[pollorder[i]]
		c = cas._chan
		sg := acquireSudog()
		sg.g = gp
		// Note: selectdone is adjusted for stack copies in stack.c:adjustsudogs
		sg.selectdone = (*uint32)(noescape(unsafe.Pointer(&done)))
		sg.elem = cas.elem
		sg.releasetime = 0
		if t0 != 0 {
			sg.releasetime = -1
		}
		sg.waitlink = gp.waiting
		gp.waiting = sg

		switch cas.kind {
		case _CaseRecv:
			c.recvq.enqueue(sg)

		case _CaseSend:
			c.sendq.enqueue(sg)
		}
	}

	// wait for someone to wake us up
	gp.param = nil
	gopark(selparkcommit, unsafe.Pointer(sel), "select", traceEvGoBlockSelect)

	// someone woke us up
	sellock(sel)
	sg = (*sudog)(gp.param)
	gp.param = nil

	// pass 3 - dequeue from unsuccessful chans
	// otherwise they stack up on quiet channels
	// record the successful case, if any.
	// We singly-linked up the SudoGs in case order, so when
	// iterating through the linked list they are in reverse order.
	cas = nil
	sglist = gp.waiting
	// Clear all elem before unlinking from gp.waiting.
	for sg1 := gp.waiting; sg1 != nil; sg1 = sg1.waitlink {
		sg1.selectdone = nil
		sg1.elem = nil
	}
	gp.waiting = nil
	for i := int(sel.ncase) - 1; i >= 0; i-- {
		k = &scases[pollorder[i]]
		if sglist.releasetime > 0 {
			k.releasetime = sglist.releasetime
		}
		if sg == sglist {
			// sg has already been dequeued by the G that woke us up.
			cas = k
		} else {
			c = k._chan
			if k.kind == _CaseSend {
				c.sendq.dequeueSudoG(sglist)
			} else {
				c.recvq.dequeueSudoG(sglist)
			}
		}
		sgnext = sglist.waitlink
		sglist.waitlink = nil
		releaseSudog(sglist)
		sglist = sgnext
	}

	if cas == nil {
		goto loop
	}

	c = cas._chan

	if c.dataqsiz > 0 {
		throw("selectgo: shouldn't happen")
	}

	if debugSelect {
		print("wait-return: sel=", sel, " c=", c, " cas=", cas, " kind=", cas.kind, "\n")
	}

	if cas.kind == _CaseRecv {
		if cas.receivedp != nil {
			*cas.receivedp = true
		}
	}

	if raceenabled {
		if cas.kind == _CaseRecv && cas.elem != nil {
			raceWriteObjectPC(c.elemtype, cas.elem, cas.pc, chanrecvpc)
		} else if cas.kind == _CaseSend {
			raceReadObjectPC(c.elemtype, cas.elem, cas.pc, chansendpc)
		}
	}

	selunlock(sel)
	goto retc

asyncrecv:
	// can receive from buffer
	if raceenabled {
		if cas.elem != nil {
			raceWriteObjectPC(c.elemtype, cas.elem, cas.pc, chanrecvpc)
		}
		raceacquire(chanbuf(c, c.recvx))
		racerelease(chanbuf(c, c.recvx))
	}
	if cas.receivedp != nil {
		*cas.receivedp = true
	}
	if cas.elem != nil {
		typedmemmove(c.elemtype, cas.elem, chanbuf(c, c.recvx))
	}
	memclr(chanbuf(c, c.recvx), uintptr(c.elemsize))
	c.recvx++
	if c.recvx == c.dataqsiz {
		c.recvx = 0
	}
	c.qcount--
	sg = c.sendq.dequeue()
	if sg != nil {
		gp = sg.g
		selunlock(sel)
		if sg.releasetime != 0 {
			sg.releasetime = cputicks()
		}
		goready(gp)
	} else {
		selunlock(sel)
	}
	goto retc

asyncsend:
	// can send to buffer
	if raceenabled {
		raceacquire(chanbuf(c, c.sendx))
		racerelease(chanbuf(c, c.sendx))
		raceReadObjectPC(c.elemtype, cas.elem, cas.pc, chansendpc)
	}
	typedmemmove(c.elemtype, chanbuf(c, c.sendx), cas.elem)
	c.sendx++
	if c.sendx == c.dataqsiz {
		c.sendx = 0
	}
	c.qcount++
	sg = c.recvq.dequeue()
	if sg != nil {
		gp = sg.g
		selunlock(sel)
		if sg.releasetime != 0 {
			sg.releasetime = cputicks()
		}
		goready(gp)
	} else {
		selunlock(sel)
	}
	goto retc

syncrecv:
	// can receive from sleeping sender (sg)
	if raceenabled {
		if cas.elem != nil {
			raceWriteObjectPC(c.elemtype, cas.elem, cas.pc, chanrecvpc)
		}
		racesync(c, sg)
	}
	selunlock(sel)
	if debugSelect {
		print("syncrecv: sel=", sel, " c=", c, "\n")
	}
	if cas.receivedp != nil {
		*cas.receivedp = true
	}
	if cas.elem != nil {
		typedmemmove(c.elemtype, cas.elem, sg.elem)
	}
	sg.elem = nil
	gp = sg.g
	gp.param = unsafe.Pointer(sg)
	if sg.releasetime != 0 {
		sg.releasetime = cputicks()
	}
	goready(gp)
	goto retc

rclose:
	// read at end of closed channel
	selunlock(sel)
	if cas.receivedp != nil {
		*cas.receivedp = false
	}
	if cas.elem != nil {
		memclr(cas.elem, uintptr(c.elemsize))
	}
	if raceenabled {
		raceacquire(unsafe.Pointer(c))
	}
	goto retc

syncsend:
	// can send to sleeping receiver (sg)
	if raceenabled {
		raceReadObjectPC(c.elemtype, cas.elem, cas.pc, chansendpc)
		racesync(c, sg)
	}
	selunlock(sel)
	if debugSelect {
		print("syncsend: sel=", sel, " c=", c, "\n")
	}
	if sg.elem != nil {
		typedmemmove(c.elemtype, sg.elem, cas.elem)
	}
	sg.elem = nil
	gp = sg.g
	gp.param = unsafe.Pointer(sg)
	if sg.releasetime != 0 {
		sg.releasetime = cputicks()
	}
	goready(gp)

retc:
	if cas.releasetime > 0 {
		blockevent(cas.releasetime-t0, 2)
	}
	return cas.pc, cas.so

sclose:
	// send on closed channel
	selunlock(sel)
	panic("send on closed channel")
}

func (c *hchan) sortkey() uintptr {
	// TODO(khr): if we have a moving garbage collector, we'll need to
	// change this function.
	return uintptr(unsafe.Pointer(c))
}

// A runtimeSelect is a single case passed to rselect.
// This must match ../reflect/value.go:/runtimeSelect
type runtimeSelect struct {
	dir selectDir
	typ unsafe.Pointer // channel type (not used here)
	ch  *hchan         // channel
	val unsafe.Pointer // ptr to data (SendDir) or ptr to receive buffer (RecvDir)
}

// These values must match ../reflect/value.go:/SelectDir.
type selectDir int

const (
	_             selectDir = iota
	selectSend              // case Chan <- Send
	selectRecv              // case <-Chan:
	selectDefault           // default
)

//go:linkname reflect_rselect reflect.rselect
func reflect_rselect(cases []runtimeSelect) (chosen int, recvOK bool) {
	// flagNoScan is safe here, because all objects are also referenced from cases.
	size := selectsize(uintptr(len(cases)))
	sel := (*_select)(mallocgc(size, nil, flagNoScan))
	newselect(sel, int64(size), int32(len(cases)))
	r := new(bool)
	for i := range cases {
		rc := &cases[i]
		switch rc.dir {
		case selectDefault:
			selectdefaultImpl(sel, uintptr(i), 0)
		case selectSend:
			if rc.ch == nil {
				break
			}
			selectsendImpl(sel, rc.ch, uintptr(i), rc.val, 0)
		case selectRecv:
			if rc.ch == nil {
				break
			}
			selectrecvImpl(sel, rc.ch, uintptr(i), rc.val, r, 0)
		}
	}

	pc, _ := selectgoImpl(sel)
	chosen = int(pc)
	recvOK = *r
	return
}

func (q *waitq) dequeueSudoG(sgp *sudog) {
	x := sgp.prev
	y := sgp.next
	if x != nil {
		if y != nil {
			// middle of queue
			x.next = y
			y.prev = x
			sgp.next = nil
			sgp.prev = nil
			return
		}
		// end of queue
		x.next = nil
		q.last = x
		sgp.prev = nil
		return
	}
	if y != nil {
		// start of queue
		y.prev = nil
		q.first = y
		sgp.next = nil
		return
	}

	// x==y==nil.  Either sgp is the only element in the queue,
	// or it has already been removed.  Use q.first to disambiguate.
	if q.first == sgp {
		q.first = nil
		q.last = nil
	}
}

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

const (
	// StackDebug == 0: no logging
	//            == 1: logging of per-stack operations
	//            == 2: logging of per-frame operations
	//            == 3: logging of per-word updates
	//            == 4: logging of per-word reads
	stackDebug       = 0
	stackFromSystem  = 0 // allocate stacks from system memory instead of the heap
	stackFaultOnFree = 0 // old stacks are mapped noaccess to detect use after free
	stackPoisonCopy  = 0 // fill stack that should not be accessed with garbage, to detect bad dereferences during copy

	stackCache = 1
)

const (
	uintptrMask = 1<<(8*ptrSize) - 1
	poisonGC    = uintptrMask & 0xf969696969696969
	poisonStack = uintptrMask & 0x6868686868686868

	// Goroutine preemption request.
	// Stored into g->stackguard0 to cause split stack check failure.
	// Must be greater than any real sp.
	// 0xfffffade in hex.
	stackPreempt = uintptrMask & -1314

	// Thread is forking.
	// Stored into g->stackguard0 to cause split stack check failure.
	// Must be greater than any real sp.
	stackFork = uintptrMask & -1234
)

// Global pool of spans that have free stacks.
// Stacks are assigned an order according to size.
//     order = log_2(size/FixedStack)
// There is a free list for each order.
// TODO: one lock per order?
var stackpool [_NumStackOrders]mspan
var stackpoolmu mutex

var stackfreequeue stack

func stackinit() {
	if _StackCacheSize&_PageMask != 0 {
		gothrow("cache size must be a multiple of page size")
	}
	for i := range stackpool {
		mSpanList_Init(&stackpool[i])
	}
}

// Allocates a stack from the free pool.  Must be called with
// stackpoolmu held.
func stackpoolalloc(order uint8) gclinkptr {
	list := &stackpool[order]
	s := list.next
	if s == list {
		// no free stacks.  Allocate another span worth.
		s = mHeap_AllocStack(&mheap_, _StackCacheSize>>_PageShift)
		if s == nil {
			gothrow("out of memory")
		}
		if s.ref != 0 {
			gothrow("bad ref")
		}
		if s.freelist.ptr() != nil {
			gothrow("bad freelist")
		}
		for i := uintptr(0); i < _StackCacheSize; i += _FixedStack << order {
			x := gclinkptr(uintptr(s.start)<<_PageShift + i)
			x.ptr().next = s.freelist
			s.freelist = x
		}
		mSpanList_Insert(list, s)
	}
	x := s.freelist
	if x.ptr() == nil {
		gothrow("span has no free stacks")
	}
	s.freelist = x.ptr().next
	s.ref++
	if s.freelist.ptr() == nil {
		// all stacks in s are allocated.
		mSpanList_Remove(s)
	}
	return x
}

// Adds stack x to the free pool.  Must be called with stackpoolmu held.
func stackpoolfree(x gclinkptr, order uint8) {
	s := mHeap_Lookup(&mheap_, (unsafe.Pointer)(x))
	if s.state != _MSpanStack {
		gothrow("freeing stack not in a stack span")
	}
	if s.freelist.ptr() == nil {
		// s will now have a free stack
		mSpanList_Insert(&stackpool[order], s)
	}
	x.ptr().next = s.freelist
	s.freelist = x
	s.ref--
	if s.ref == 0 {
		// span is completely free - return to heap
		mSpanList_Remove(s)
		s.freelist = 0
		mHeap_FreeStack(&mheap_, s)
	}
}

// stackcacherefill/stackcacherelease implement a global pool of stack segments.
// The pool is required to prevent unlimited growth of per-thread caches.
func stackcacherefill(c *mcache, order uint8) {
	if stackDebug >= 1 {
		print("stackcacherefill order=", order, "\n")
	}

	// Grab some stacks from the global cache.
	// Grab half of the allowed capacity (to prevent thrashing).
	var list gclinkptr
	var size uintptr
	lock(&stackpoolmu)
	for size < _StackCacheSize/2 {
		x := stackpoolalloc(order)
		x.ptr().next = list
		list = x
		size += _FixedStack << order
	}
	unlock(&stackpoolmu)
	c.stackcache[order].list = list
	c.stackcache[order].size = size
}

func stackcacherelease(c *mcache, order uint8) {
	if stackDebug >= 1 {
		print("stackcacherelease order=", order, "\n")
	}
	x := c.stackcache[order].list
	size := c.stackcache[order].size
	lock(&stackpoolmu)
	for size > _StackCacheSize/2 {
		y := x.ptr().next
		stackpoolfree(x, order)
		x = y
		size -= _FixedStack << order
	}
	unlock(&stackpoolmu)
	c.stackcache[order].list = x
	c.stackcache[order].size = size
}

func stackcache_clear(c *mcache) {
	if stackDebug >= 1 {
		print("stackcache clear\n")
	}
	lock(&stackpoolmu)
	for order := uint8(0); order < _NumStackOrders; order++ {
		x := c.stackcache[order].list
		for x.ptr() != nil {
			y := x.ptr().next
			stackpoolfree(x, order)
			x = y
		}
		c.stackcache[order].list = 0
		c.stackcache[order].size = 0
	}
	unlock(&stackpoolmu)
}

func stackalloc(n uint32) stack {
	// Stackalloc must be called on scheduler stack, so that we
	// never try to grow the stack during the code that stackalloc runs.
	// Doing so would cause a deadlock (issue 1547).
	thisg := getg()
	if thisg != thisg.m.g0 {
		gothrow("stackalloc not on scheduler stack")
	}
	if n&(n-1) != 0 {
		gothrow("stack size not a power of 2")
	}
	if stackDebug >= 1 {
		print("stackalloc ", n, "\n")
	}

	if debug.efence != 0 || stackFromSystem != 0 {
		v := sysAlloc(round(uintptr(n), _PageSize), &memstats.stacks_sys)
		if v == nil {
			gothrow("out of memory (stackalloc)")
		}
		return stack{uintptr(v), uintptr(v) + uintptr(n)}
	}

	// Small stacks are allocated with a fixed-size free-list allocator.
	// If we need a stack of a bigger size, we fall back on allocating
	// a dedicated span.
	var v unsafe.Pointer
	if stackCache != 0 && n < _FixedStack<<_NumStackOrders && n < _StackCacheSize {
		order := uint8(0)
		n2 := n
		for n2 > _FixedStack {
			order++
			n2 >>= 1
		}
		var x gclinkptr
		c := thisg.m.mcache
		if c == nil || thisg.m.gcing != 0 || thisg.m.helpgc != 0 {
			// c == nil can happen in the guts of exitsyscall or
			// procresize. Just get a stack from the global pool.
			// Also don't touch stackcache during gc
			// as it's flushed concurrently.
			lock(&stackpoolmu)
			x = stackpoolalloc(order)
			unlock(&stackpoolmu)
		} else {
			x = c.stackcache[order].list
			if x.ptr() == nil {
				stackcacherefill(c, order)
				x = c.stackcache[order].list
			}
			c.stackcache[order].list = x.ptr().next
			c.stackcache[order].size -= uintptr(n)
		}
		v = (unsafe.Pointer)(x)
	} else {
		s := mHeap_AllocStack(&mheap_, round(uintptr(n), _PageSize)>>_PageShift)
		if s == nil {
			gothrow("out of memory")
		}
		v = (unsafe.Pointer)(s.start << _PageShift)
	}

	if raceenabled {
		racemalloc(v, uintptr(n))
	}
	if stackDebug >= 1 {
		print("  allocated ", v, "\n")
	}
	return stack{uintptr(v), uintptr(v) + uintptr(n)}
}

func stackfree(stk stack) {
	gp := getg()
	n := stk.hi - stk.lo
	v := (unsafe.Pointer)(stk.lo)
	if n&(n-1) != 0 {
		gothrow("stack not a power of 2")
	}
	if stackDebug >= 1 {
		println("stackfree", v, n)
		memclr(v, n) // for testing, clobber stack data
	}
	if debug.efence != 0 || stackFromSystem != 0 {
		if debug.efence != 0 || stackFaultOnFree != 0 {
			sysFault(v, n)
		} else {
			sysFree(v, n, &memstats.stacks_sys)
		}
		return
	}
	if stackCache != 0 && n < _FixedStack<<_NumStackOrders && n < _StackCacheSize {
		order := uint8(0)
		n2 := n
		for n2 > _FixedStack {
			order++
			n2 >>= 1
		}
		x := gclinkptr(v)
		c := gp.m.mcache
		if c == nil || gp.m.gcing != 0 || gp.m.helpgc != 0 {
			lock(&stackpoolmu)
			stackpoolfree(x, order)
			unlock(&stackpoolmu)
		} else {
			if c.stackcache[order].size >= _StackCacheSize {
				stackcacherelease(c, order)
			}
			x.ptr().next = c.stackcache[order].list
			c.stackcache[order].list = x
			c.stackcache[order].size += n
		}
	} else {
		s := mHeap_Lookup(&mheap_, v)
		if s.state != _MSpanStack {
			println(hex(s.start<<_PageShift), v)
			gothrow("bad span state")
		}
		mHeap_FreeStack(&mheap_, s)
	}
}

var maxstacksize uintptr = 1 << 20 // enough until runtime.main sets it for real

var mapnames = []string{
	_BitsDead:    "---",
	_BitsScalar:  "scalar",
	_BitsPointer: "ptr",
}

// Stack frame layout
//
// (x86)
// +------------------+
// | args from caller |
// +------------------+ <- frame->argp
// |  return address  |
// +------------------+ <- frame->varp
// |     locals       |
// +------------------+
// |  args to callee  |
// +------------------+ <- frame->sp
//
// (arm)
// +------------------+
// | args from caller |
// +------------------+ <- frame->argp
// | caller's retaddr |
// +------------------+ <- frame->varp
// |     locals       |
// +------------------+
// |  args to callee  |
// +------------------+
// |  return address  |
// +------------------+ <- frame->sp

type adjustinfo struct {
	old   stack
	delta uintptr // ptr distance from old to new stack (newbase - oldbase)
}

// Adjustpointer checks whether *vpp is in the old stack described by adjinfo.
// If so, it rewrites *vpp to point into the new stack.
func adjustpointer(adjinfo *adjustinfo, vpp unsafe.Pointer) {
	pp := (*unsafe.Pointer)(vpp)
	p := *pp
	if stackDebug >= 4 {
		print("        ", pp, ":", p, "\n")
	}
	if adjinfo.old.lo <= uintptr(p) && uintptr(p) < adjinfo.old.hi {
		*pp = add(p, adjinfo.delta)
		if stackDebug >= 3 {
			print("        adjust ptr ", pp, ":", p, " -> ", *pp, "\n")
		}
	}
}

type gobitvector struct {
	n        uintptr
	bytedata []uint8
}

func gobv(bv bitvector) gobitvector {
	return gobitvector{
		uintptr(bv.n),
		(*[1 << 30]byte)(unsafe.Pointer(bv.bytedata))[:(bv.n+7)/8],
	}
}

func ptrbits(bv *gobitvector, i uintptr) uint8 {
	return (bv.bytedata[i/4] >> ((i & 3) * 2)) & 3
}

// bv describes the memory starting at address scanp.
// Adjust any pointers contained therein.
func adjustpointers(scanp unsafe.Pointer, cbv *bitvector, adjinfo *adjustinfo, f *_func) {
	bv := gobv(*cbv)
	minp := adjinfo.old.lo
	maxp := adjinfo.old.hi
	delta := adjinfo.delta
	num := uintptr(bv.n / _BitsPerPointer)
	for i := uintptr(0); i < num; i++ {
		if stackDebug >= 4 {
			print("        ", add(scanp, i*ptrSize), ":", mapnames[ptrbits(&bv, i)], ":", hex(*(*uintptr)(add(scanp, i*ptrSize))), " # ", i, " ", bv.bytedata[i/4], "\n")
		}
		switch ptrbits(&bv, i) {
		default:
			gothrow("unexpected pointer bits")
		case _BitsDead:
			if debug.gcdead != 0 {
				*(*unsafe.Pointer)(add(scanp, i*ptrSize)) = unsafe.Pointer(uintptr(poisonStack))
			}
		case _BitsScalar:
			// ok
		case _BitsPointer:
			p := *(*unsafe.Pointer)(add(scanp, i*ptrSize))
			up := uintptr(p)
			if f != nil && 0 < up && up < _PageSize && invalidptr != 0 || up == poisonGC || up == poisonStack {
				// Looks like a junk value in a pointer slot.
				// Live analysis wrong?
				getg().m.traceback = 2
				print("runtime: bad pointer in frame ", gofuncname(f), " at ", add(scanp, i*ptrSize), ": ", p, "\n")
				gothrow("invalid stack pointer")
			}
			if minp <= up && up < maxp {
				if stackDebug >= 3 {
					print("adjust ptr ", p, " ", gofuncname(f), "\n")
				}
				*(*unsafe.Pointer)(add(scanp, i*ptrSize)) = unsafe.Pointer(up + delta)
			}
		}
	}
}

// Note: the argument/return area is adjusted by the callee.
func adjustframe(frame *stkframe, arg unsafe.Pointer) bool {
	adjinfo := (*adjustinfo)(arg)
	targetpc := frame.continpc
	if targetpc == 0 {
		// Frame is dead.
		return true
	}
	f := frame.fn
	if stackDebug >= 2 {
		print("    adjusting ", funcname(f), " frame=[", hex(frame.sp), ",", hex(frame.fp), "] pc=", hex(frame.pc), " continpc=", hex(frame.continpc), "\n")
	}
	if f.entry == systemstack_switchPC {
		// A special routine at the bottom of stack of a goroutine that does an systemstack call.
		// We will allow it to be copied even though we don't
		// have full GC info for it (because it is written in asm).
		return true
	}
	if targetpc != f.entry {
		targetpc--
	}
	pcdata := pcdatavalue(f, _PCDATA_StackMapIndex, targetpc)
	if pcdata == -1 {
		pcdata = 0 // in prologue
	}

	// Adjust local variables if stack frame has been allocated.
	size := frame.varp - frame.sp
	var minsize uintptr
	if thechar != '6' && thechar != '8' {
		minsize = ptrSize
	} else {
		minsize = 0
	}
	if size > minsize {
		var bv bitvector
		stackmap := (*stackmap)(funcdata(f, _FUNCDATA_LocalsPointerMaps))
		if stackmap == nil || stackmap.n <= 0 {
			print("runtime: frame ", funcname(f), " untyped locals ", hex(frame.varp-size), "+", hex(size), "\n")
			gothrow("missing stackmap")
		}
		// Locals bitmap information, scan just the pointers in locals.
		if pcdata < 0 || pcdata >= stackmap.n {
			// don't know where we are
			print("runtime: pcdata is ", pcdata, " and ", stackmap.n, " locals stack map entries for ", funcname(f), " (targetpc=", targetpc, ")\n")
			gothrow("bad symbol table")
		}
		bv = stackmapdata(stackmap, pcdata)
		size = (uintptr(bv.n) * ptrSize) / _BitsPerPointer
		if stackDebug >= 3 {
			print("      locals ", pcdata, "/", stackmap.n, " ", size/ptrSize, " words ", bv.bytedata, "\n")
		}
		adjustpointers(unsafe.Pointer(frame.varp-size), &bv, adjinfo, f)
	}

	// Adjust arguments.
	if frame.arglen > 0 {
		var bv bitvector
		if frame.argmap != nil {
			bv = *frame.argmap
		} else {
			stackmap := (*stackmap)(funcdata(f, _FUNCDATA_ArgsPointerMaps))
			if stackmap == nil || stackmap.n <= 0 {
				print("runtime: frame ", funcname(f), " untyped args ", frame.argp, "+", uintptr(frame.arglen), "\n")
				gothrow("missing stackmap")
			}
			if pcdata < 0 || pcdata >= stackmap.n {
				// don't know where we are
				print("runtime: pcdata is ", pcdata, " and ", stackmap.n, " args stack map entries for ", funcname(f), " (targetpc=", targetpc, ")\n")
				gothrow("bad symbol table")
			}
			bv = stackmapdata(stackmap, pcdata)
		}
		if stackDebug >= 3 {
			print("      args\n")
		}
		adjustpointers(unsafe.Pointer(frame.argp), &bv, adjinfo, nil)
	}
	return true
}

func adjustctxt(gp *g, adjinfo *adjustinfo) {
	adjustpointer(adjinfo, (unsafe.Pointer)(&gp.sched.ctxt))
}

func adjustdefers(gp *g, adjinfo *adjustinfo) {
	// Adjust defer argument blocks the same way we adjust active stack frames.
	tracebackdefers(gp, adjustframe, noescape(unsafe.Pointer(adjinfo)))

	// Adjust pointers in the Defer structs.
	// Defer structs themselves are never on the stack.
	for d := gp._defer; d != nil; d = d.link {
		adjustpointer(adjinfo, (unsafe.Pointer)(&d.fn))
		adjustpointer(adjinfo, (unsafe.Pointer)(&d.argp))
		adjustpointer(adjinfo, (unsafe.Pointer)(&d._panic))
	}
}

func adjustpanics(gp *g, adjinfo *adjustinfo) {
	// Panics are on stack and already adjusted.
	// Update pointer to head of list in G.
	adjustpointer(adjinfo, (unsafe.Pointer)(&gp._panic))
}

func adjustsudogs(gp *g, adjinfo *adjustinfo) {
	// the data elements pointed to by a SudoG structure
	// might be in the stack.
	for s := gp.waiting; s != nil; s = s.waitlink {
		adjustpointer(adjinfo, (unsafe.Pointer)(&s.elem))
		adjustpointer(adjinfo, (unsafe.Pointer)(&s.selectdone))
	}
}

func fillstack(stk stack, b byte) {
	for p := stk.lo; p < stk.hi; p++ {
		*(*byte)(unsafe.Pointer(p)) = b
	}
}

// Copies gp's stack to a new stack of a different size.
// Caller must have changed gp status to Gcopystack.
func copystack(gp *g, newsize uintptr) {
	if gp.syscallsp != 0 {
		gothrow("stack growth not allowed in system call")
	}
	old := gp.stack
	if old.lo == 0 {
		gothrow("nil stackbase")
	}
	used := old.hi - gp.sched.sp

	// allocate new stack
	new := stackalloc(uint32(newsize))
	if stackPoisonCopy != 0 {
		fillstack(new, 0xfd)
	}
	if stackDebug >= 1 {
		print("copystack gp=", gp, " [", hex(old.lo), " ", hex(old.hi-used), " ", hex(old.hi), "]/", old.hi-old.lo, " -> [", hex(new.lo), " ", hex(new.hi-used), " ", hex(new.hi), "]/", newsize, "\n")
	}

	// adjust pointers in the to-be-copied frames
	var adjinfo adjustinfo
	adjinfo.old = old
	adjinfo.delta = new.hi - old.hi
	gentraceback(^uintptr(0), ^uintptr(0), 0, gp, 0, nil, 0x7fffffff, adjustframe, noescape(unsafe.Pointer(&adjinfo)), 0)

	// adjust other miscellaneous things that have pointers into stacks.
	adjustctxt(gp, &adjinfo)
	adjustdefers(gp, &adjinfo)
	adjustpanics(gp, &adjinfo)
	adjustsudogs(gp, &adjinfo)

	// copy the stack to the new location
	if stackPoisonCopy != 0 {
		fillstack(new, 0xfb)
	}
	memmove(unsafe.Pointer(new.hi-used), unsafe.Pointer(old.hi-used), used)

	// Swap out old stack for new one
	gp.stack = new
	gp.stackguard0 = new.lo + _StackGuard // NOTE: might clobber a preempt request
	gp.sched.sp = new.hi - used

	// free old stack
	if stackPoisonCopy != 0 {
		fillstack(old, 0xfc)
	}
	if newsize > old.hi-old.lo {
		// growing, free stack immediately
		stackfree(old)
	} else {
		// shrinking, queue up free operation.  We can't actually free the stack
		// just yet because we might run into the following situation:
		// 1) GC starts, scans a SudoG but does not yet mark the SudoG.elem pointer
		// 2) The stack that pointer points to is shrunk
		// 3) The old stack is freed
		// 4) The containing span is marked free
		// 5) GC attempts to mark the SudoG.elem pointer.  The marking fails because
		//    the pointer looks like a pointer into a free span.
		// By not freeing, we prevent step #4 until GC is done.
		lock(&stackpoolmu)
		*(*stack)(unsafe.Pointer(old.lo)) = stackfreequeue
		stackfreequeue = old
		unlock(&stackpoolmu)
	}
}

// round x up to a power of 2.
func round2(x int32) int32 {
	s := uint(0)
	for 1<<s < x {
		s++
	}
	return 1 << s
}

// Called from runtime·morestack when more stack is needed.
// Allocate larger stack and relocate to new stack.
// Stack growth is multiplicative, for constant amortized cost.
//
// g->atomicstatus will be Grunning or Gscanrunning upon entry.
// If the GC is trying to stop this g then it will set preemptscan to true.
func newstack() {
	thisg := getg()
	// TODO: double check all gp. shouldn't be getg().
	if thisg.m.morebuf.g.stackguard0 == stackFork {
		gothrow("stack growth after fork")
	}
	if thisg.m.morebuf.g != thisg.m.curg {
		print("runtime: newstack called from g=", thisg.m.morebuf.g, "\n"+"\tm=", thisg.m, " m->curg=", thisg.m.curg, " m->g0=", thisg.m.g0, " m->gsignal=", thisg.m.gsignal, "\n")
		morebuf := thisg.m.morebuf
		traceback(morebuf.pc, morebuf.sp, morebuf.lr, morebuf.g)
		gothrow("runtime: wrong goroutine in newstack")
	}
	if thisg.m.curg.throwsplit {
		gp := thisg.m.curg
		// Update syscallsp, syscallpc in case traceback uses them.
		morebuf := thisg.m.morebuf
		gp.syscallsp = morebuf.sp
		gp.syscallpc = morebuf.pc
		print("runtime: newstack sp=", hex(gp.sched.sp), " stack=[", hex(gp.stack.lo), ", ", hex(gp.stack.hi), "]\n",
			"\tmorebuf={pc:", hex(morebuf.pc), " sp:", hex(morebuf.sp), " lr:", hex(morebuf.lr), "}\n",
			"\tsched={pc:", hex(gp.sched.pc), " sp:", hex(gp.sched.sp), " lr:", hex(gp.sched.lr), " ctxt:", gp.sched.ctxt, "}\n")
		gothrow("runtime: stack split at bad time")
	}

	// The goroutine must be executing in order to call newstack,
	// so it must be Grunning or Gscanrunning.

	gp := thisg.m.curg
	morebuf := thisg.m.morebuf
	thisg.m.morebuf.pc = 0
	thisg.m.morebuf.lr = 0
	thisg.m.morebuf.sp = 0
	thisg.m.morebuf.g = nil

	casgstatus(gp, _Grunning, _Gwaiting)
	gp.waitreason = "stack growth"

	rewindmorestack(&gp.sched)

	if gp.stack.lo == 0 {
		gothrow("missing stack in newstack")
	}
	sp := gp.sched.sp
	if thechar == '6' || thechar == '8' {
		// The call to morestack cost a word.
		sp -= ptrSize
	}
	if stackDebug >= 1 || sp < gp.stack.lo {
		print("runtime: newstack sp=", hex(sp), " stack=[", hex(gp.stack.lo), ", ", hex(gp.stack.hi), "]\n",
			"\tmorebuf={pc:", hex(morebuf.pc), " sp:", hex(morebuf.sp), " lr:", hex(morebuf.lr), "}\n",
			"\tsched={pc:", hex(gp.sched.pc), " sp:", hex(gp.sched.sp), " lr:", hex(gp.sched.lr), " ctxt:", gp.sched.ctxt, "}\n")
	}
	if sp < gp.stack.lo {
		print("runtime: gp=", gp, ", gp->status=", hex(readgstatus(gp)), "\n ")
		print("runtime: split stack overflow: ", hex(sp), " < ", hex(gp.stack.lo), "\n")
		gothrow("runtime: split stack overflow")
	}

	if gp.sched.ctxt != nil {
		// morestack wrote sched.ctxt on its way in here,
		// without a write barrier. Run the write barrier now.
		// It is not possible to be preempted between then
		// and now, so it's okay.
		writebarrierptr_nostore((*uintptr)(unsafe.Pointer(&gp.sched.ctxt)), uintptr(gp.sched.ctxt))
	}

	if gp.stackguard0 == stackPreempt {
		if gp == thisg.m.g0 {
			gothrow("runtime: preempt g0")
		}
		if thisg.m.p == nil && thisg.m.locks == 0 {
			gothrow("runtime: g is running but p is not")
		}
		if gp.preemptscan {
			for !castogscanstatus(gp, _Gwaiting, _Gscanwaiting) {
				// Likely to be racing with the GC as it sees a _Gwaiting and does the stack scan.
				// If so this stack will be scanned twice which does not change correctness.
			}
			gcphasework(gp)
			casfrom_Gscanstatus(gp, _Gscanwaiting, _Gwaiting)
			casgstatus(gp, _Gwaiting, _Grunning)
			gp.stackguard0 = gp.stack.lo + _StackGuard
			gp.preempt = false
			gp.preemptscan = false // Tells the GC premption was successful.
			gogo(&gp.sched)        // never return
		}

		// Be conservative about where we preempt.
		// We are interested in preempting user Go code, not runtime code.
		if thisg.m.locks != 0 || thisg.m.mallocing != 0 || thisg.m.gcing != 0 || thisg.m.p.status != _Prunning {
			// Let the goroutine keep running for now.
			// gp->preempt is set, so it will be preempted next time.
			gp.stackguard0 = gp.stack.lo + _StackGuard
			casgstatus(gp, _Gwaiting, _Grunning)
			gogo(&gp.sched) // never return
		}

		// Act like goroutine called runtime.Gosched.
		casgstatus(gp, _Gwaiting, _Grunning)
		gosched_m(gp) // never return
	}

	// Allocate a bigger segment and move the stack.
	oldsize := int(gp.stack.hi - gp.stack.lo)
	newsize := oldsize * 2
	if uintptr(newsize) > maxstacksize {
		print("runtime: goroutine stack exceeds ", maxstacksize, "-byte limit\n")
		gothrow("stack overflow")
	}

	casgstatus(gp, _Gwaiting, _Gcopystack)

	// The concurrent GC will not scan the stack while we are doing the copy since
	// the gp is in a Gcopystack status.
	copystack(gp, uintptr(newsize))
	if stackDebug >= 1 {
		print("stack grow done\n")
	}
	casgstatus(gp, _Gcopystack, _Grunning)
	gogo(&gp.sched)
}

//go:nosplit
func nilfunc() {
	*(*uint8)(nil) = 0
}

// adjust Gobuf as if it executed a call to fn
// and then did an immediate gosave.
func gostartcallfn(gobuf *gobuf, fv *funcval) {
	var fn unsafe.Pointer
	if fv != nil {
		fn = (unsafe.Pointer)(fv.fn)
	} else {
		fn = unsafe.Pointer(funcPC(nilfunc))
	}
	gostartcall(gobuf, fn, (unsafe.Pointer)(fv))
}

// Maybe shrink the stack being used by gp.
// Called at garbage collection time.
func shrinkstack(gp *g) {
	if readgstatus(gp) == _Gdead {
		if gp.stack.lo != 0 {
			// Free whole stack - it will get reallocated
			// if G is used again.
			stackfree(gp.stack)
			gp.stack.lo = 0
			gp.stack.hi = 0
		}
		return
	}
	if gp.stack.lo == 0 {
		gothrow("missing stack in shrinkstack")
	}

	oldsize := gp.stack.hi - gp.stack.lo
	newsize := oldsize / 2
	if newsize < _FixedStack {
		return // don't shrink below the minimum-sized stack
	}
	used := gp.stack.hi - gp.sched.sp
	if used >= oldsize/4 {
		return // still using at least 1/4 of the segment.
	}

	// We can't copy the stack if we're in a syscall.
	// The syscall might have pointers into the stack.
	if gp.syscallsp != 0 {
		return
	}
	if goos_windows != 0 && gp.m != nil && gp.m.libcallsp != 0 {
		return
	}

	if stackDebug > 0 {
		print("shrinking stack ", oldsize, "->", newsize, "\n")
	}

	oldstatus := casgcopystack(gp)
	copystack(gp, newsize)
	casgstatus(gp, _Gcopystack, oldstatus)
}

// Do any delayed stack freeing that was queued up during GC.
func shrinkfinish() {
	lock(&stackpoolmu)
	s := stackfreequeue
	stackfreequeue = stack{}
	unlock(&stackpoolmu)
	for s.lo != 0 {
		t := *(*stack)(unsafe.Pointer(s.lo))
		stackfree(s)
		s = t
	}
}

//go:nosplit
func morestackc() {
	systemstack(func() {
		gothrow("attempt to execute C code on Go stack")
	})
}

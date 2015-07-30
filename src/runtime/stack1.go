// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

const (
	// stackDebug == 0: no logging
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

// List of stack spans to be freed at the end of GC. Protected by
// stackpoolmu.
var stackFreeQueue mspan

// Cached value of haveexperiment("framepointer")
var framepointer_enabled bool

func stackinit() {
	if _StackCacheSize&_PageMask != 0 {
		throw("cache size must be a multiple of page size")
	}
	for i := range stackpool {
		mSpanList_Init(&stackpool[i])
	}
	mSpanList_Init(&stackFreeQueue)
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
			throw("out of memory")
		}
		if s.ref != 0 {
			throw("bad ref")
		}
		if s.freelist.ptr() != nil {
			throw("bad freelist")
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
		throw("span has no free stacks")
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
		throw("freeing stack not in a stack span")
	}
	if s.freelist.ptr() == nil {
		// s will now have a free stack
		mSpanList_Insert(&stackpool[order], s)
	}
	x.ptr().next = s.freelist
	s.freelist = x
	s.ref--
	if gcphase == _GCoff && s.ref == 0 {
		// Span is completely free. Return it to the heap
		// immediately if we're sweeping.
		//
		// If GC is active, we delay the free until the end of
		// GC to avoid the following type of situation:
		//
		// 1) GC starts, scans a SudoG but does not yet mark the SudoG.elem pointer
		// 2) The stack that pointer points to is copied
		// 3) The old stack is freed
		// 4) The containing span is marked free
		// 5) GC attempts to mark the SudoG.elem pointer. The
		//    marking fails because the pointer looks like a
		//    pointer into a free span.
		//
		// By not freeing, we prevent step #4 until GC is done.
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

func stackalloc(n uint32) (stack, []stkbar) {
	// Stackalloc must be called on scheduler stack, so that we
	// never try to grow the stack during the code that stackalloc runs.
	// Doing so would cause a deadlock (issue 1547).
	thisg := getg()
	if thisg != thisg.m.g0 {
		throw("stackalloc not on scheduler stack")
	}
	if n&(n-1) != 0 {
		throw("stack size not a power of 2")
	}
	if stackDebug >= 1 {
		print("stackalloc ", n, "\n")
	}

	// Compute the size of stack barrier array.
	maxstkbar := gcMaxStackBarriers(int(n))
	nstkbar := unsafe.Sizeof(stkbar{}) * uintptr(maxstkbar)

	if debug.efence != 0 || stackFromSystem != 0 {
		v := sysAlloc(round(uintptr(n), _PageSize), &memstats.stacks_sys)
		if v == nil {
			throw("out of memory (stackalloc)")
		}
		top := uintptr(n) - nstkbar
		stkbarSlice := slice{add(v, top), 0, maxstkbar}
		return stack{uintptr(v), uintptr(v) + top}, *(*[]stkbar)(unsafe.Pointer(&stkbarSlice))
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
		if c == nil || thisg.m.preemptoff != "" || thisg.m.helpgc != 0 {
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
			throw("out of memory")
		}
		v = (unsafe.Pointer)(s.start << _PageShift)
	}

	if raceenabled {
		racemalloc(v, uintptr(n))
	}
	if stackDebug >= 1 {
		print("  allocated ", v, "\n")
	}
	top := uintptr(n) - nstkbar
	stkbarSlice := slice{add(v, top), 0, maxstkbar}
	return stack{uintptr(v), uintptr(v) + top}, *(*[]stkbar)(unsafe.Pointer(&stkbarSlice))
}

func stackfree(stk stack, n uintptr) {
	gp := getg()
	v := (unsafe.Pointer)(stk.lo)
	if n&(n-1) != 0 {
		throw("stack not a power of 2")
	}
	if stk.lo+n < stk.hi {
		throw("bad stack size")
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
		if c == nil || gp.m.preemptoff != "" || gp.m.helpgc != 0 {
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
			throw("bad span state")
		}
		if gcphase == _GCoff {
			// Free the stack immediately if we're
			// sweeping.
			mHeap_FreeStack(&mheap_, s)
		} else {
			// Otherwise, add it to a list of stack spans
			// to be freed at the end of GC.
			//
			// TODO(austin): Make it possible to re-use
			// these spans as stacks, like we do for small
			// stack spans. (See issue #11466.)
			lock(&stackpoolmu)
			mSpanList_Insert(&stackFreeQueue, s)
			unlock(&stackpoolmu)
		}
	}
}

var maxstacksize uintptr = 1 << 20 // enough until runtime.main sets it for real

var ptrnames = []string{
	0: "scalar",
	1: "ptr",
}

// Stack frame layout
//
// (x86)
// +------------------+
// | args from caller |
// +------------------+ <- frame->argp
// |  return address  |
// +------------------+
// |  caller's BP (*) | (*) if framepointer_enabled && varp < sp
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

// Information from the compiler about the layout of stack frames.
type bitvector struct {
	n        int32 // # of bits
	bytedata *uint8
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

func ptrbit(bv *gobitvector, i uintptr) uint8 {
	return (bv.bytedata[i/8] >> (i % 8)) & 1
}

// bv describes the memory starting at address scanp.
// Adjust any pointers contained therein.
func adjustpointers(scanp unsafe.Pointer, cbv *bitvector, adjinfo *adjustinfo, f *_func) {
	bv := gobv(*cbv)
	minp := adjinfo.old.lo
	maxp := adjinfo.old.hi
	delta := adjinfo.delta
	num := uintptr(bv.n)
	for i := uintptr(0); i < num; i++ {
		if stackDebug >= 4 {
			print("        ", add(scanp, i*ptrSize), ":", ptrnames[ptrbit(&bv, i)], ":", hex(*(*uintptr)(add(scanp, i*ptrSize))), " # ", i, " ", bv.bytedata[i/8], "\n")
		}
		if ptrbit(&bv, i) == 1 {
			pp := (*uintptr)(add(scanp, i*ptrSize))
			p := *pp
			if f != nil && 0 < p && p < _PageSize && debug.invalidptr != 0 || p == poisonStack {
				// Looks like a junk value in a pointer slot.
				// Live analysis wrong?
				getg().m.traceback = 2
				print("runtime: bad pointer in frame ", funcname(f), " at ", pp, ": ", hex(p), "\n")
				throw("invalid stack pointer")
			}
			if minp <= p && p < maxp {
				if stackDebug >= 3 {
					print("adjust ptr ", p, " ", funcname(f), "\n")
				}
				*pp = p + delta
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
	switch thechar {
	case '6', '8':
		minsize = 0
	case '7':
		minsize = spAlign
	default:
		minsize = ptrSize
	}
	if size > minsize {
		var bv bitvector
		stackmap := (*stackmap)(funcdata(f, _FUNCDATA_LocalsPointerMaps))
		if stackmap == nil || stackmap.n <= 0 {
			print("runtime: frame ", funcname(f), " untyped locals ", hex(frame.varp-size), "+", hex(size), "\n")
			throw("missing stackmap")
		}
		// Locals bitmap information, scan just the pointers in locals.
		if pcdata < 0 || pcdata >= stackmap.n {
			// don't know where we are
			print("runtime: pcdata is ", pcdata, " and ", stackmap.n, " locals stack map entries for ", funcname(f), " (targetpc=", targetpc, ")\n")
			throw("bad symbol table")
		}
		bv = stackmapdata(stackmap, pcdata)
		size = uintptr(bv.n) * ptrSize
		if stackDebug >= 3 {
			print("      locals ", pcdata, "/", stackmap.n, " ", size/ptrSize, " words ", bv.bytedata, "\n")
		}
		adjustpointers(unsafe.Pointer(frame.varp-size), &bv, adjinfo, f)
	}

	// Adjust saved base pointer if there is one.
	if thechar == '6' && frame.argp-frame.varp == 2*regSize {
		if !framepointer_enabled {
			print("runtime: found space for saved base pointer, but no framepointer experiment\n")
			print("argp=", hex(frame.argp), " varp=", hex(frame.varp), "\n")
			throw("bad frame layout")
		}
		if stackDebug >= 3 {
			print("      saved bp\n")
		}
		adjustpointer(adjinfo, unsafe.Pointer(frame.varp))
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
				throw("missing stackmap")
			}
			if pcdata < 0 || pcdata >= stackmap.n {
				// don't know where we are
				print("runtime: pcdata is ", pcdata, " and ", stackmap.n, " args stack map entries for ", funcname(f), " (targetpc=", targetpc, ")\n")
				throw("bad symbol table")
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
		adjustpointer(adjinfo, (unsafe.Pointer)(&d.sp))
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

func adjuststkbar(gp *g, adjinfo *adjustinfo) {
	for i := int(gp.stkbarPos); i < len(gp.stkbar); i++ {
		adjustpointer(adjinfo, (unsafe.Pointer)(&gp.stkbar[i].savedLRPtr))
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
		throw("stack growth not allowed in system call")
	}
	old := gp.stack
	if old.lo == 0 {
		throw("nil stackbase")
	}
	used := old.hi - gp.sched.sp

	// allocate new stack
	new, newstkbar := stackalloc(uint32(newsize))
	if stackPoisonCopy != 0 {
		fillstack(new, 0xfd)
	}
	if stackDebug >= 1 {
		print("copystack gp=", gp, " [", hex(old.lo), " ", hex(old.hi-used), " ", hex(old.hi), "]/", gp.stackAlloc, " -> [", hex(new.lo), " ", hex(new.hi-used), " ", hex(new.hi), "]/", newsize, "\n")
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
	adjuststkbar(gp, &adjinfo)

	// copy the stack to the new location
	if stackPoisonCopy != 0 {
		fillstack(new, 0xfb)
	}
	memmove(unsafe.Pointer(new.hi-used), unsafe.Pointer(old.hi-used), used)

	// copy old stack barriers to new stack barrier array
	newstkbar = newstkbar[:len(gp.stkbar)]
	copy(newstkbar, gp.stkbar)

	// Swap out old stack for new one
	gp.stack = new
	gp.stackguard0 = new.lo + _StackGuard // NOTE: might clobber a preempt request
	gp.sched.sp = new.hi - used
	oldsize := gp.stackAlloc
	gp.stackAlloc = newsize
	gp.stkbar = newstkbar

	// free old stack
	if stackPoisonCopy != 0 {
		fillstack(old, 0xfc)
	}
	stackfree(old, oldsize)
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
	if thisg.m.morebuf.g.ptr().stackguard0 == stackFork {
		throw("stack growth after fork")
	}
	if thisg.m.morebuf.g.ptr() != thisg.m.curg {
		print("runtime: newstack called from g=", thisg.m.morebuf.g, "\n"+"\tm=", thisg.m, " m->curg=", thisg.m.curg, " m->g0=", thisg.m.g0, " m->gsignal=", thisg.m.gsignal, "\n")
		morebuf := thisg.m.morebuf
		traceback(morebuf.pc, morebuf.sp, morebuf.lr, morebuf.g.ptr())
		throw("runtime: wrong goroutine in newstack")
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

		traceback(morebuf.pc, morebuf.sp, morebuf.lr, gp)
		throw("runtime: stack split at bad time")
	}

	gp := thisg.m.curg
	morebuf := thisg.m.morebuf
	thisg.m.morebuf.pc = 0
	thisg.m.morebuf.lr = 0
	thisg.m.morebuf.sp = 0
	thisg.m.morebuf.g = 0
	rewindmorestack(&gp.sched)

	// NOTE: stackguard0 may change underfoot, if another thread
	// is about to try to preempt gp. Read it just once and use that same
	// value now and below.
	preempt := atomicloaduintptr(&gp.stackguard0) == stackPreempt

	// Be conservative about where we preempt.
	// We are interested in preempting user Go code, not runtime code.
	// If we're holding locks, mallocing, or preemption is disabled, don't
	// preempt.
	// This check is very early in newstack so that even the status change
	// from Grunning to Gwaiting and back doesn't happen in this case.
	// That status change by itself can be viewed as a small preemption,
	// because the GC might change Gwaiting to Gscanwaiting, and then
	// this goroutine has to wait for the GC to finish before continuing.
	// If the GC is in some way dependent on this goroutine (for example,
	// it needs a lock held by the goroutine), that small preemption turns
	// into a real deadlock.
	if preempt {
		if thisg.m.locks != 0 || thisg.m.mallocing != 0 || thisg.m.preemptoff != "" || thisg.m.p.ptr().status != _Prunning {
			// Let the goroutine keep running for now.
			// gp->preempt is set, so it will be preempted next time.
			gp.stackguard0 = gp.stack.lo + _StackGuard
			gogo(&gp.sched) // never return
		}
	}

	// The goroutine must be executing in order to call newstack,
	// so it must be Grunning (or Gscanrunning).
	casgstatus(gp, _Grunning, _Gwaiting)
	gp.waitreason = "stack growth"

	if gp.stack.lo == 0 {
		throw("missing stack in newstack")
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
		throw("runtime: split stack overflow")
	}

	if gp.sched.ctxt != nil {
		// morestack wrote sched.ctxt on its way in here,
		// without a write barrier. Run the write barrier now.
		// It is not possible to be preempted between then
		// and now, so it's okay.
		writebarrierptr_nostore((*uintptr)(unsafe.Pointer(&gp.sched.ctxt)), uintptr(gp.sched.ctxt))
	}

	if preempt {
		if gp == thisg.m.g0 {
			throw("runtime: preempt g0")
		}
		if thisg.m.p == 0 && thisg.m.locks == 0 {
			throw("runtime: g is running but p is not")
		}
		if gp.preemptscan {
			for !castogscanstatus(gp, _Gwaiting, _Gscanwaiting) {
				// Likely to be racing with the GC as
				// it sees a _Gwaiting and does the
				// stack scan. If so, gcworkdone will
				// be set and gcphasework will simply
				// return.
			}
			if !gp.gcscandone {
				scanstack(gp)
				gp.gcscandone = true
			}
			gp.preemptscan = false
			gp.preempt = false
			casfrom_Gscanstatus(gp, _Gscanwaiting, _Gwaiting)
			casgstatus(gp, _Gwaiting, _Grunning)
			gp.stackguard0 = gp.stack.lo + _StackGuard
			gogo(&gp.sched) // never return
		}

		// Act like goroutine called runtime.Gosched.
		casgstatus(gp, _Gwaiting, _Grunning)
		gopreempt_m(gp) // never return
	}

	// Allocate a bigger segment and move the stack.
	oldsize := int(gp.stackAlloc)
	newsize := oldsize * 2
	if uintptr(newsize) > maxstacksize {
		print("runtime: goroutine stack exceeds ", maxstacksize, "-byte limit\n")
		throw("stack overflow")
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
			stackfree(gp.stack, gp.stackAlloc)
			gp.stack.lo = 0
			gp.stack.hi = 0
			gp.stkbar = nil
			gp.stkbarPos = 0
		}
		return
	}
	if gp.stack.lo == 0 {
		throw("missing stack in shrinkstack")
	}

	if debug.gcshrinkstackoff > 0 {
		return
	}

	oldsize := gp.stackAlloc
	newsize := oldsize / 2
	// Don't shrink the allocation below the minimum-sized stack
	// allocation.
	if newsize < _FixedStack {
		return
	}
	// Compute how much of the stack is currently in use and only
	// shrink the stack if gp is using less than a quarter of its
	// current stack. The currently used stack includes everything
	// down to the SP plus the stack guard space that ensures
	// there's room for nosplit functions.
	avail := gp.stack.hi - gp.stack.lo
	if used := gp.stack.hi - gp.sched.sp + _StackLimit; used >= avail/4 {
		return
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

// freeStackSpans frees unused stack spans at the end of GC.
func freeStackSpans() {
	lock(&stackpoolmu)

	// Scan stack pools for empty stack spans.
	for order := range stackpool {
		list := &stackpool[order]
		for s := list.next; s != list; {
			next := s.next
			if s.ref == 0 {
				mSpanList_Remove(s)
				s.freelist = 0
				mHeap_FreeStack(&mheap_, s)
			}
			s = next
		}
	}

	// Free queued stack spans.
	for stackFreeQueue.next != &stackFreeQueue {
		s := stackFreeQueue.next
		mSpanList_Remove(s)
		mHeap_FreeStack(&mheap_, s)
	}

	unlock(&stackpoolmu)
}

//go:nosplit
func morestackc() {
	systemstack(func() {
		throw("attempt to execute C code on Go stack")
	})
}

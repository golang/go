// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

/*
 * defined constants
 */
const (
	// G status
	//
	// If you add to this list, add to the list
	// of "okay during garbage collection" status
	// in mgc0.c too.
	_Gidle            = iota // 0
	_Grunnable               // 1 runnable and on a run queue
	_Grunning                // 2
	_Gsyscall                // 3
	_Gwaiting                // 4
	_Gmoribund_unused        // 5 currently unused, but hardcoded in gdb scripts
	_Gdead                   // 6
	_Genqueue                // 7 Only the Gscanenqueue is used.
	_Gcopystack              // 8 in this state when newstack is moving the stack
	// the following encode that the GC is scanning the stack and what to do when it is done
	_Gscan = 0x1000 // atomicstatus&~Gscan = the non-scan state,
	// _Gscanidle =     _Gscan + _Gidle,      // Not used. Gidle only used with newly malloced gs
	_Gscanrunnable = _Gscan + _Grunnable //  0x1001 When scanning complets make Grunnable (it is already on run queue)
	_Gscanrunning  = _Gscan + _Grunning  //  0x1002 Used to tell preemption newstack routine to scan preempted stack.
	_Gscansyscall  = _Gscan + _Gsyscall  //  0x1003 When scanning completes make is Gsyscall
	_Gscanwaiting  = _Gscan + _Gwaiting  //  0x1004 When scanning completes make it Gwaiting
	// _Gscanmoribund_unused,               //  not possible
	// _Gscandead,                          //  not possible
	_Gscanenqueue = _Gscan + _Genqueue //  When scanning completes make it Grunnable and put on runqueue
)

const (
	// P status
	_Pidle = iota
	_Prunning
	_Psyscall
	_Pgcstop
	_Pdead
)

// The next line makes 'go generate' write the zgen_*.go files with
// per-OS and per-arch information, including constants
// named goos_$GOOS and goarch_$GOARCH for every
// known GOOS and GOARCH. The constant is 1 on the
// current system, 0 otherwise; multiplying by them is
// useful for defining GOOS- or GOARCH-specific constants.
//go:generate go run gengoos.go

type mutex struct {
	// Futex-based impl treats it as uint32 key,
	// while sema-based impl as M* waitm.
	// Used to be a union, but unions break precise GC.
	key uintptr
}

type note struct {
	// Futex-based impl treats it as uint32 key,
	// while sema-based impl as M* waitm.
	// Used to be a union, but unions break precise GC.
	key uintptr
}

type _string struct {
	str *byte
	len int
}

type funcval struct {
	fn uintptr
	// variable-size, fn-specific data here
}

type iface struct {
	tab  *itab
	data unsafe.Pointer
}

type eface struct {
	_type *_type
	data  unsafe.Pointer
}

type slice struct {
	array *byte // actual data
	len   uint  // number of elements
	cap   uint  // allocated number of elements
}

// A guintptr holds a goroutine pointer, but typed as a uintptr
// to bypass write barriers. It is used in the Gobuf goroutine state.
//
// The Gobuf.g goroutine pointer is almost always updated by assembly code.
// In one of the few places it is updated by Go code - func save - it must be
// treated as a uintptr to avoid a write barrier being emitted at a bad time.
// Instead of figuring out how to emit the write barriers missing in the
// assembly manipulation, we change the type of the field to uintptr,
// so that it does not require write barriers at all.
//
// Goroutine structs are published in the allg list and never freed.
// That will keep the goroutine structs from being collected.
// There is never a time that Gobuf.g's contain the only references
// to a goroutine: the publishing of the goroutine in allg comes first.
// Goroutine pointers are also kept in non-GC-visible places like TLS,
// so I can't see them ever moving. If we did want to start moving data
// in the GC, we'd need to allocate the goroutine structs from an
// alternate arena. Using guintptr doesn't make that problem any worse.
type guintptr uintptr

func (gp guintptr) ptr() *g {
	return (*g)(unsafe.Pointer(gp))
}

type gobuf struct {
	// The offsets of sp, pc, and g are known to (hard-coded in) libmach.
	sp   uintptr
	pc   uintptr
	g    guintptr
	ctxt unsafe.Pointer // this has to be a pointer so that gc scans it
	ret  uintreg
	lr   uintptr
}

// Known to compiler.
// Changes here must also be made in src/cmd/gc/select.c's selecttype.
type sudog struct {
	g           *g
	selectdone  *uint32
	next        *sudog
	prev        *sudog
	elem        unsafe.Pointer // data element
	releasetime int64
	nrelease    int32  // -1 for acquire
	waitlink    *sudog // g.waiting list
}

type gcstats struct {
	// the struct must consist of only uint64's,
	// because it is casted to uint64[].
	nhandoff    uint64
	nhandoffcnt uint64
	nprocyield  uint64
	nosyield    uint64
	nsleep      uint64
}

type libcall struct {
	fn   uintptr
	n    uintptr // number of parameters
	args uintptr // parameters
	r1   uintptr // return values
	r2   uintptr
	err  uintptr // error number
}

// describes how to handle callback
type wincallbackcontext struct {
	gobody       unsafe.Pointer // go function to call
	argsize      uintptr        // callback arguments size (in bytes)
	restorestack uintptr        // adjust stack on return by (in bytes) (386 only)
	cleanstack   bool
}

// Stack describes a Go execution stack.
// The bounds of the stack are exactly [lo, hi),
// with no implicit data structures on either side.
type stack struct {
	lo uintptr
	hi uintptr
}

type g struct {
	// Stack parameters.
	// stack describes the actual stack memory: [stack.lo, stack.hi).
	// stackguard0 is the stack pointer compared in the Go stack growth prologue.
	// It is stack.lo+StackGuard normally, but can be StackPreempt to trigger a preemption.
	// stackguard1 is the stack pointer compared in the C stack growth prologue.
	// It is stack.lo+StackGuard on g0 and gsignal stacks.
	// It is ~0 on other goroutine stacks, to trigger a call to morestackc (and crash).
	stack       stack   // offset known to runtime/cgo
	stackguard0 uintptr // offset known to liblink
	stackguard1 uintptr // offset known to liblink

	_panic       *_panic // innermost panic - offset known to liblink
	_defer       *_defer // innermost defer
	sched        gobuf
	syscallsp    uintptr        // if status==gsyscall, syscallsp = sched.sp to use during gc
	syscallpc    uintptr        // if status==gsyscall, syscallpc = sched.pc to use during gc
	param        unsafe.Pointer // passed parameter on wakeup
	atomicstatus uint32
	goid         int64
	waitsince    int64  // approx time when the g become blocked
	waitreason   string // if status==gwaiting
	schedlink    *g
	issystem     bool // do not output in stack dump, ignore in deadlock detector
	preempt      bool // preemption signal, duplicates stackguard0 = stackpreempt
	paniconfault bool // panic (instead of crash) on unexpected fault address
	preemptscan  bool // preempted g does scan for gc
	gcworkdone   bool // debug: cleared at begining of gc work phase cycle, set by gcphasework, tested at end of cycle
	gcscanvalid  bool // false at start of gc cycle, true if G has not run since last scan
	throwsplit   bool // must not split stack
	raceignore   int8 // ignore race detection events
	m            *m   // for debuggers, but offset not hard-coded
	lockedm      *m
	sig          uint32
	writebuf     []byte
	sigcode0     uintptr
	sigcode1     uintptr
	sigpc        uintptr
	gopc         uintptr // pc of go statement that created this goroutine
	startpc      uintptr // pc of goroutine function
	racectx      uintptr
	waiting      *sudog // sudog structures this g is waiting on (that have a valid elem ptr)
}

type mts struct {
	tv_sec  int64
	tv_nsec int64
}

type mscratch struct {
	v [6]uintptr
}

type m struct {
	g0      *g    // goroutine with scheduling stack
	morebuf gobuf // gobuf arg to morestack

	// Fields not known to debuggers.
	procid        uint64         // for debuggers, but offset not hard-coded
	gsignal       *g             // signal-handling g
	tls           [4]uintptr     // thread-local storage (for x86 extern register)
	mstartfn      unsafe.Pointer // todo go func()
	curg          *g             // current running goroutine
	caughtsig     *g             // goroutine running during fatal signal
	p             *p             // attached p for executing go code (nil if not executing go code)
	nextp         *p
	id            int32
	mallocing     int32
	throwing      int32
	gcing         int32
	locks         int32
	softfloat     int32
	dying         int32
	profilehz     int32
	helpgc        int32
	spinning      bool // m is out of work and is actively looking for work
	blocked       bool // m is blocked on a note
	inwb          bool // m is executing a write barrier
	printlock     int8
	fastrand      uint32
	ncgocall      uint64 // number of cgo calls in total
	ncgo          int32  // number of cgo calls currently in progress
	cgomal        *cgomal
	park          note
	alllink       *m // on allm
	schedlink     *m
	machport      uint32 // return address for mach ipc (os x)
	mcache        *mcache
	lockedg       *g
	createstack   [32]uintptr // stack that created this thread.
	freglo        [16]uint32  // d[i] lsb and f[i]
	freghi        [16]uint32  // d[i] msb and f[i+16]
	fflag         uint32      // floating point compare flags
	locked        uint32      // tracking for lockosthread
	nextwaitm     *m          // next m waiting for lock
	waitsema      uintptr     // semaphore for parking on locks
	waitsemacount uint32
	waitsemalock  uint32
	gcstats       gcstats
	needextram    bool
	traceback     uint8
	waitunlockf   unsafe.Pointer // todo go func(*g, unsafe.pointer) bool
	waitlock      unsafe.Pointer
	waittraceev   byte
	syscalltick   uint32
	//#ifdef GOOS_windows
	thread uintptr // thread handle
	// these are here because they are too large to be on the stack
	// of low-level NOSPLIT functions.
	libcall   libcall
	libcallpc uintptr // for cpu profiler
	libcallsp uintptr
	libcallg  *g
	//#endif
	//#ifdef GOOS_solaris
	perrno *int32 // pointer to tls errno
	// these are here because they are too large to be on the stack
	// of low-level NOSPLIT functions.
	//LibCall	libcall;
	ts      mts
	scratch mscratch
	//#endif
	//#ifdef GOOS_plan9
	notesig *int8
	errstr  *byte
	//#endif
}

type p struct {
	lock mutex

	id          int32
	status      uint32 // one of pidle/prunning/...
	link        *p
	schedtick   uint32 // incremented on every scheduler call
	syscalltick uint32 // incremented on every system call
	m           *m     // back-link to associated m (nil if idle)
	mcache      *mcache
	deferpool   [5]*_defer // pool of available defer structs of different sizes (see panic.c)

	// Cache of goroutine ids, amortizes accesses to runtime·sched.goidgen.
	goidcache    uint64
	goidcacheend uint64

	// Queue of runnable goroutines.
	runqhead uint32
	runqtail uint32
	runq     [256]*g

	// Available G's (status == Gdead)
	gfree    *g
	gfreecnt int32

	tracebuf *traceBuf

	pad [64]byte
}

const (
	// The max value of GOMAXPROCS.
	// There are no fundamental restrictions on the value.
	_MaxGomaxprocs = 1 << 8
)

type schedt struct {
	lock mutex

	goidgen uint64

	midle        *m    // idle m's waiting for work
	nmidle       int32 // number of idle m's waiting for work
	nmidlelocked int32 // number of locked m's waiting for work
	mcount       int32 // number of m's that have been created
	maxmcount    int32 // maximum number of m's allowed (or die)

	pidle      *p // idle p's
	npidle     uint32
	nmspinning uint32

	// Global runnable queue.
	runqhead *g
	runqtail *g
	runqsize int32

	// Global cache of dead G's.
	gflock mutex
	gfree  *g
	ngfree int32

	gcwaiting  uint32 // gc is waiting to run
	stopwait   int32
	stopnote   note
	sysmonwait uint32
	sysmonnote note
	lastpoll   uint64

	profilehz int32 // cpu profiling rate
}

// The m->locked word holds two pieces of state counting active calls to LockOSThread/lockOSThread.
// The low bit (LockExternal) is a boolean reporting whether any LockOSThread call is active.
// External locks are not recursive; a second lock is silently ignored.
// The upper bits of m->lockedcount record the nesting depth of calls to lockOSThread
// (counting up by LockInternal), popped by unlockOSThread (counting down by LockInternal).
// Internal locks can be recursive. For instance, a lock for cgo can occur while the main
// goroutine is holding the lock during the initialization phase.
const (
	_LockExternal = 1
	_LockInternal = 2
)

type sigtabtt struct {
	flags int32
	name  *int8
}

const (
	_SigNotify   = 1 << 0 // let signal.Notify have signal, even if from kernel
	_SigKill     = 1 << 1 // if signal.Notify doesn't take it, exit quietly
	_SigThrow    = 1 << 2 // if signal.Notify doesn't take it, exit loudly
	_SigPanic    = 1 << 3 // if the signal is from the kernel, panic
	_SigDefault  = 1 << 4 // if the signal isn't explicitly requested, don't monitor it
	_SigHandling = 1 << 5 // our signal handler is registered
	_SigIgnored  = 1 << 6 // the signal was ignored before we registered for it
	_SigGoExit   = 1 << 7 // cause all runtime procs to exit (only used on Plan 9).
	_SigSetStack = 1 << 8 // add SA_ONSTACK to libc handler
)

// Layout of in-memory per-function information prepared by linker
// See http://golang.org/s/go12symtab.
// Keep in sync with linker and with ../../libmach/sym.c
// and with package debug/gosym and with symtab.go in package runtime.
type _func struct {
	entry   uintptr // start pc
	nameoff int32   // function name

	args  int32 // in/out args size
	frame int32 // legacy frame size; use pcsp if possible

	pcsp      int32
	pcfile    int32
	pcln      int32
	npcdata   int32
	nfuncdata int32
}

// layout of Itab known to compilers
// allocated in non-garbage-collected memory
type itab struct {
	inter  *interfacetype
	_type  *_type
	link   *itab
	bad    int32
	unused int32
	fun    [1]uintptr // variable sized
}

// Lock-free stack node.
// // Also known to export_test.go.
type lfnode struct {
	next    uint64
	pushcnt uintptr
}

// Parallel for descriptor.
type parfor struct {
	body    unsafe.Pointer // go func(*parfor, uint32), executed for each element
	done    uint32         // number of idle threads
	nthr    uint32         // total number of threads
	nthrmax uint32         // maximum number of threads
	thrseq  uint32         // thread id sequencer
	cnt     uint32         // iteration space [0, cnt)
	ctx     unsafe.Pointer // arbitrary user context
	wait    bool           // if true, wait while all threads finish processing,
	// otherwise parfor may return while other threads are still working
	thr *parforthread // array of thread descriptors
	pad uint32        // to align parforthread.pos for 64-bit atomic operations
	// stats
	nsteal     uint64
	nstealcnt  uint64
	nprocyield uint64
	nosyield   uint64
	nsleep     uint64
}

// Track memory allocated by code not written in Go during a cgo call,
// so that the garbage collector can see them.
type cgomal struct {
	next  *cgomal
	alloc unsafe.Pointer
}

// Indicates to write barrier and sychronization task to preform.
const (
	_GCoff             = iota // GC not running, write barrier disabled
	_GCquiesce                // unused state
	_GCstw                    // unused state
	_GCscan                   // GC collecting roots into workbufs, write barrier disabled
	_GCmark                   // GC marking from workbufs, write barrier ENABLED
	_GCmarktermination        // GC mark termination: allocate black, P's help GC, write barrier ENABLED
	_GCsweep                  // GC mark completed; sweeping in background, write barrier disabled
)

type forcegcstate struct {
	lock mutex
	g    *g
	idle uint32
}

var gcphase uint32

/*
 * known to compiler
 */
const (
	_Structrnd = regSize
)

// startup_random_data holds random bytes initialized at startup.  These come from
// the ELF AT_RANDOM auxiliary vector (vdso_linux_amd64.go or os_linux_386.go).
var startupRandomData []byte

// extendRandom extends the random numbers in r[:n] to the whole slice r.
// Treats n<0 as n==0.
func extendRandom(r []byte, n int) {
	if n < 0 {
		n = 0
	}
	for n < len(r) {
		// Extend random bits using hash function & time seed
		w := n
		if w > 16 {
			w = 16
		}
		h := memhash(unsafe.Pointer(&r[n-w]), uintptr(nanotime()), uintptr(w))
		for i := 0; i < ptrSize && n < len(r); i++ {
			r[n] = byte(h)
			n++
			h >>= 8
		}
	}
}

/*
 * deferred subroutine calls
 */
type _defer struct {
	siz     int32
	started bool
	sp      uintptr // sp at time of defer
	pc      uintptr
	fn      *funcval
	_panic  *_panic // panic that is running defer
	link    *_defer
}

/*
 * panics
 */
type _panic struct {
	argp      unsafe.Pointer // pointer to arguments of deferred call run during panic; cannot move - known to liblink
	arg       interface{}    // argument to panic
	link      *_panic        // link to earlier panic
	recovered bool           // whether this panic is over
	aborted   bool           // the panic was aborted
}

/*
 * stack traces
 */

type stkframe struct {
	fn       *_func     // function being run
	pc       uintptr    // program counter within fn
	continpc uintptr    // program counter where execution can continue, or 0 if not
	lr       uintptr    // program counter at caller aka link register
	sp       uintptr    // stack pointer at pc
	fp       uintptr    // stack pointer at caller aka frame pointer
	varp     uintptr    // top of local variables
	argp     uintptr    // pointer to function arguments
	arglen   uintptr    // number of bytes at argp
	argmap   *bitvector // force use of this argmap
}

const (
	_TraceRuntimeFrames = 1 << 0 // include frames for internal runtime functions.
	_TraceTrap          = 1 << 1 // the initial PC, SP are from a trap, not a return PC from a call
)

const (
	// The maximum number of frames we print for a traceback
	_TracebackMaxFrames = 100
)

var (
	emptystring string
	allg        **g
	allglen     uintptr
	lastg       *g
	allm        *m
	allp        [_MaxGomaxprocs + 1]*p
	gomaxprocs  int32
	needextram  uint32
	panicking   uint32
	goos        *int8
	ncpu        int32
	iscgo       bool
	cpuid_ecx   uint32
	cpuid_edx   uint32
	signote     note
	forcegc     forcegcstate
	sched       schedt
	newprocs    int32
)

/*
 * mutual exclusion locks.  in the uncontended case,
 * as fast as spin locks (just a few user-level instructions),
 * but on the contention path they sleep in the kernel.
 * a zeroed Mutex is unlocked (no need to initialize each lock).
 */

/*
 * sleep and wakeup on one-time events.
 * before any calls to notesleep or notewakeup,
 * must call noteclear to initialize the Note.
 * then, exactly one thread can call notesleep
 * and exactly one thread can call notewakeup (once).
 * once notewakeup has been called, the notesleep
 * will return.  future notesleep will return immediately.
 * subsequent noteclear must be called only after
 * previous notesleep has returned, e.g. it's disallowed
 * to call noteclear straight after notewakeup.
 *
 * notetsleep is like notesleep but wakes up after
 * a given number of nanoseconds even if the event
 * has not yet happened.  if a goroutine uses notetsleep to
 * wake up early, it must wait to call noteclear until it
 * can be sure that no other goroutine is calling
 * notewakeup.
 *
 * notesleep/notetsleep are generally called on g0,
 * notetsleepg is similar to notetsleep but is called on user g.
 */
// bool	runtime·notetsleep(Note*, int64);  // false - timeout
// bool	runtime·notetsleepg(Note*, int64);  // false - timeout

/*
 * Lock-free stack.
 * Initialize uint64 head to 0, compare with 0 to test for emptiness.
 * The stack does not keep pointers to nodes,
 * so they can be garbage collected if there are no other pointers to nodes.
 */

/*
 * Parallel for over [0, n).
 * body() is executed for each iteration.
 * nthr - total number of worker threads.
 * ctx - arbitrary user context.
 * if wait=true, threads return from parfor() when all work is done;
 * otherwise, threads can return while other threads are still finishing processing.
 */

// for mmap, we only pass the lower 32 bits of file offset to the
// assembly routine; the higher bits (if required), should be provided
// by the assembly routine as 0.

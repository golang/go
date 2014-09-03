// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * basic types
 */
typedef	signed char		int8;
typedef	unsigned char		uint8;
typedef	signed short		int16;
typedef	unsigned short		uint16;
typedef	signed int		int32;
typedef	unsigned int		uint32;
typedef	signed long long int	int64;
typedef	unsigned long long int	uint64;
typedef	float			float32;
typedef	double			float64;

#ifdef _64BIT
typedef	uint64		uintptr;
typedef	int64		intptr;
typedef	int64		intgo; // Go's int
typedef	uint64		uintgo; // Go's uint
#else
typedef	uint32		uintptr;
typedef	int32		intptr;
typedef	int32		intgo; // Go's int
typedef	uint32		uintgo; // Go's uint
#endif

#ifdef _64BITREG
typedef	uint64		uintreg;
#else
typedef	uint32		uintreg;
#endif

/*
 * get rid of C types
 * the / / / forces a syntax error immediately,
 * which will show "last name: XXunsigned".
 */
#define	unsigned		XXunsigned / / /
#define	signed			XXsigned / / /
#define	char			XXchar / / /
#define	short			XXshort / / /
#define	int			XXint / / /
#define	long			XXlong / / /
#define	float			XXfloat / / /
#define	double			XXdouble / / /

/*
 * defined types
 */
typedef	uint8			bool;
typedef	uint8			byte;
typedef	struct	Func		Func;
typedef	struct	G		G;
typedef	struct	Gobuf		Gobuf;
typedef	struct	SudoG		SudoG;
typedef	struct	Mutex		Mutex;
typedef	struct	M		M;
typedef	struct	P		P;
typedef	struct	Note		Note;
typedef	struct	Slice		Slice;
typedef	struct	Stktop		Stktop;
typedef	struct	String		String;
typedef	struct	FuncVal		FuncVal;
typedef	struct	SigTab		SigTab;
typedef	struct	MCache		MCache;
typedef	struct	FixAlloc	FixAlloc;
typedef	struct	Iface		Iface;
typedef	struct	Itab		Itab;
typedef	struct	InterfaceType	InterfaceType;
typedef	struct	Eface		Eface;
typedef	struct	Type		Type;
typedef	struct	PtrType		PtrType;
typedef	struct	ChanType		ChanType;
typedef	struct	MapType		MapType;
typedef	struct	Defer		Defer;
typedef	struct	Panic		Panic;
typedef	struct	Hmap		Hmap;
typedef	struct	Hiter			Hiter;
typedef	struct	Hchan		Hchan;
typedef	struct	Complex64	Complex64;
typedef	struct	Complex128	Complex128;
typedef	struct	LibCall		LibCall;
typedef	struct	WinCallbackContext	WinCallbackContext;
typedef	struct	GCStats		GCStats;
typedef	struct	LFNode		LFNode;
typedef	struct	ParFor		ParFor;
typedef	struct	ParForThread	ParForThread;
typedef	struct	CgoMal		CgoMal;
typedef	struct	PollDesc	PollDesc;
typedef	struct	DebugVars	DebugVars;
typedef struct	ForceGCState	ForceGCState;

/*
 * Per-CPU declaration.
 *
 * "extern register" is a special storage class implemented by 6c, 8c, etc.
 * On the ARM, it is an actual register; elsewhere it is a slot in thread-
 * local storage indexed by a pseudo-register TLS. See zasmhdr in
 * src/cmd/dist/buildruntime.c for details, and be aware that the linker may
 * make further OS-specific changes to the compiler's output. For example,
 * 6l/linux rewrites 0(TLS) as -8(FS).
 *
 * Every C file linked into a Go program must include runtime.h so that the
 * C compiler (6c, 8c, etc.) knows to avoid other uses of these dedicated
 * registers. The Go compiler (6g, 8g, etc.) knows to avoid them.
 */
extern	register	G*	g;

/*
 * defined constants
 */
enum
{
	// G status
	//
	// If you add to this list, add to the list
	// of "okay during garbage collection" status
	// in mgc0.c too.
	Gidle,                                 // 0
	Grunnable,                             // 1 runnable and on a run queue
	Grunning,                              // 2
	Gsyscall,                              // 3
	Gwaiting,                              // 4
	Gmoribund_unused,                      // 5 currently unused, but hardcoded in gdb scripts
	Gdead,                                 // 6
	Genqueue,                              // 7 Only the Gscanenqueue is used.
	Gcopystack,                            // 8 in this state when newstack is moving the stack
	// the following encode that the GC is scanning the stack and what to do when it is done 
	Gscan = 0x1000,                        // atomicstatus&~Gscan = the non-scan state,
	// Gscanidle =     Gscan + Gidle,      // Not used. Gidle only used with newly malloced gs
	Gscanrunnable = Gscan + Grunnable,     //  0x1001 When scanning complets make Grunnable (it is already on run queue)
	Gscanrunning =  Gscan + Grunning,      //  0x1002 Used to tell preemption newstack routine to scan preempted stack.
	Gscansyscall =  Gscan + Gsyscall,      //  0x1003 When scanning completes make is Gsyscall
	Gscanwaiting =  Gscan + Gwaiting,      //  0x1004 When scanning completes make it Gwaiting
	// Gscanmoribund_unused,               //  not possible
	// Gscandead,                          //  not possible
	Gscanenqueue = Gscan + Genqueue,       //  When scanning completes make it Grunnable and put on runqueue
};
enum
{
	// P status
	Pidle,
	Prunning,
	Psyscall,
	Pgcstop,
	Pdead,
};
enum
{
	true	= 1,
	false	= 0,
};
enum
{
	PtrSize = sizeof(void*),
};
/*
 * structures
 */
struct	Mutex
{
	// Futex-based impl treats it as uint32 key,
	// while sema-based impl as M* waitm.
	// Used to be a union, but unions break precise GC.
	uintptr	key;
};
struct	Note
{
	// Futex-based impl treats it as uint32 key,
	// while sema-based impl as M* waitm.
	// Used to be a union, but unions break precise GC.
	uintptr	key;
};
struct String
{
	byte*	str;
	intgo	len;
};
struct FuncVal
{
	void	(*fn)(void);
	// variable-size, fn-specific data here
};
struct Iface
{
	Itab*	tab;
	void*	data;
};
struct Eface
{
	Type*	type;
	void*	data;
};
struct Complex64
{
	float32	real;
	float32	imag;
};
struct Complex128
{
	float64	real;
	float64	imag;
};

struct	Slice
{				// must not move anything
	byte*	array;		// actual data
	uintgo	len;		// number of elements
	uintgo	cap;		// allocated number of elements
};
struct	Gobuf
{
	// The offsets of sp, pc, and g are known to (hard-coded in) libmach.
	uintptr	sp;
	uintptr	pc;
	G*	g;
	void*	ctxt; // this has to be a pointer so that GC scans it
	uintreg	ret;
	uintptr	lr;
};
// Known to compiler.
// Changes here must also be made in src/cmd/gc/select.c's selecttype.
struct	SudoG
{
	G*	g;
	uint32*	selectdone;
	SudoG*	next;
	SudoG*	prev;
	void*	elem;		// data element
	int64	releasetime;
	int32	nrelease;	// -1 for acquire
	SudoG*	waitlink;	// G.waiting list
};
struct	GCStats
{
	// the struct must consist of only uint64's,
	// because it is casted to uint64[].
	uint64	nhandoff;
	uint64	nhandoffcnt;
	uint64	nprocyield;
	uint64	nosyield;
	uint64	nsleep;
};

struct	LibCall
{
	void*	fn;
	uintptr	n;	// number of parameters
	void*	args;	// parameters
	uintptr	r1;	// return values
	uintptr	r2;
	uintptr	err;	// error number
};

// describes how to handle callback
struct	WinCallbackContext
{
	void*	gobody;		// Go function to call
	uintptr	argsize;	// callback arguments size (in bytes)
	uintptr	restorestack;	// adjust stack on return by (in bytes) (386 only)
	bool	cleanstack;
};

struct	G
{
	// stackguard0 can be set to StackPreempt as opposed to stackguard
	uintptr	stackguard0;	// cannot move - also known to linker, libmach, runtime/cgo
	uintptr	stackbase;	// cannot move - also known to libmach, runtime/cgo
	uint32	panicwrap;	// cannot move - also known to linker
	Defer*	defer;
	Panic*	panic;
	Gobuf	sched;
	uintptr	syscallstack;	// if status==Gsyscall, syscallstack = stackbase to use during gc
	uintptr	syscallsp;	// if status==Gsyscall, syscallsp = sched.sp to use during gc
	uintptr	syscallpc;	// if status==Gsyscall, syscallpc = sched.pc to use during gc
	uintptr	syscallguard;	// if status==Gsyscall, syscallguard = stackguard to use during gc
	uintptr	stackguard;	// same as stackguard0, but not set to StackPreempt
	uintptr	stack0;
	uintptr	stacksize;
	void*	param;		// passed parameter on wakeup
	uint32	atomicstatus;
	int64	goid;
	int64	waitsince;	// approx time when the G become blocked
	String	waitreason;	// if status==Gwaiting
	G*	schedlink;
	bool	ispanic;
	bool	issystem;	// do not output in stack dump, ignore in deadlock detector
	bool	preempt;	// preemption signal, duplicates stackguard0 = StackPreempt
	bool	paniconfault;	// panic (instead of crash) on unexpected fault address
	bool    preemptscan;    // preempted g does scan for GC
	bool    scancheck;      // debug: cleared at begining of scan cycle, set by scan, tested at end of cycle
	int8	raceignore;	// ignore race detection events
	M*	m;		// for debuggers, but offset not hard-coded
	M*	lockedm;
	int32	sig;
	int32	writenbuf;
	Slice	writebuf;
	uintptr	sigcode0;
	uintptr	sigcode1;
	uintptr	sigpc;
	uintptr	gopc;		// pc of go statement that created this goroutine
	uintptr	racectx;
	SudoG   *waiting;	// sudog structures this G is waiting on (that have a valid elem ptr)
	uintptr	end[];
};

struct	M
{
	G*	g0;		// goroutine with scheduling stack
	void*	moreargp;	// argument pointer for more stack
	Gobuf	morebuf;	// gobuf arg to morestack

	// Fields not known to debuggers.
	uint32	moreframesize;	// size arguments to morestack
	uint32	moreargsize;	// known by amd64 asm to follow moreframesize
	uintreg	cret;		// return value from C
	uint64	procid;		// for debuggers, but offset not hard-coded
	G*	gsignal;	// signal-handling G
	uintptr	tls[4];		// thread-local storage (for x86 extern register)
	void	(*mstartfn)(void);
	G*	curg;		// current running goroutine
	G*	caughtsig;	// goroutine running during fatal signal
	P*	p;		// attached P for executing Go code (nil if not executing Go code)
	P*	nextp;
	int32	id;
	int32	mallocing;
	int32	throwing;
	int32	gcing;
	int32	locks;
	int32	softfloat;
	int32	dying;
	int32	profilehz;
	int32	helpgc;
	bool	spinning;	// M is out of work and is actively looking for work
	bool	blocked;	// M is blocked on a Note
	uint32	fastrand;
	uint64	ncgocall;	// number of cgo calls in total
	int32	ncgo;		// number of cgo calls currently in progress
	CgoMal*	cgomal;
	Note	park;
	M*	alllink;	// on allm
	M*	schedlink;
	uint32	machport;	// Return address for Mach IPC (OS X)
	MCache*	mcache;
	G*	lockedg;
	uintptr	createstack[32];// Stack that created this thread.
	uint32	freglo[16];	// D[i] lsb and F[i]
	uint32	freghi[16];	// D[i] msb and F[i+16]
	uint32	fflag;		// floating point compare flags
	uint32	locked;		// tracking for LockOSThread
	M*	nextwaitm;	// next M waiting for lock
	uintptr	waitsema;	// semaphore for parking on locks
	uint32	waitsemacount;
	uint32	waitsemalock;
	GCStats	gcstats;
	bool	needextram;
	uint8	traceback;
	bool	(*waitunlockf)(G*, void*);
	void*	waitlock;
	uintptr	forkstackguard;
	uintptr scalararg[4];	// scalar argument/return for mcall
	void*   ptrarg[4];	// pointer argument/return for mcall
#ifdef GOOS_windows
	void*	thread;		// thread handle
	// these are here because they are too large to be on the stack
	// of low-level NOSPLIT functions.
	LibCall	libcall;
	uintptr	libcallpc;	// for cpu profiler
	uintptr	libcallsp;
	G*	libcallg;
#endif
#ifdef GOOS_solaris
	int32*	perrno; 	// pointer to TLS errno
	// these are here because they are too large to be on the stack
	// of low-level NOSPLIT functions.
	LibCall	libcall;
	struct {
		int64	tv_sec;
		int64	tv_nsec;
	} ts;
	struct {
		uintptr v[6];
	} scratch;
#endif
#ifdef GOOS_plan9
	int8*	notesig;
	byte*	errstr;
#endif
	uintptr	end[];
};

struct P
{
	Mutex	lock;

	int32	id;
	uint32	status;		// one of Pidle/Prunning/...
	P*	link;
	uint32	schedtick;	// incremented on every scheduler call
	uint32	syscalltick;	// incremented on every system call
	M*	m;		// back-link to associated M (nil if idle)
	MCache*	mcache;
	Defer*	deferpool[5];	// pool of available Defer structs of different sizes (see panic.c)

	// Cache of goroutine ids, amortizes accesses to runtime·sched.goidgen.
	uint64	goidcache;
	uint64	goidcacheend;

	// Queue of runnable goroutines.
	uint32	runqhead;
	uint32	runqtail;
	G*	runq[256];

	// Available G's (status == Gdead)
	G*	gfree;
	int32	gfreecnt;

	byte	pad[64];
};

enum {
	// The max value of GOMAXPROCS.
	// There are no fundamental restrictions on the value.
	MaxGomaxprocs = 1<<8,
};

// The m->locked word holds two pieces of state counting active calls to LockOSThread/lockOSThread.
// The low bit (LockExternal) is a boolean reporting whether any LockOSThread call is active.
// External locks are not recursive; a second lock is silently ignored.
// The upper bits of m->lockedcount record the nesting depth of calls to lockOSThread
// (counting up by LockInternal), popped by unlockOSThread (counting down by LockInternal).
// Internal locks can be recursive. For instance, a lock for cgo can occur while the main
// goroutine is holding the lock during the initialization phase.
enum
{
	LockExternal = 1,
	LockInternal = 2,
};

struct	Stktop
{
	// The offsets of these fields are known to (hard-coded in) libmach.
	uintptr	stackguard;
	uintptr	stackbase;
	Gobuf	gobuf;
	uint32	argsize;
	uint32	panicwrap;

	uint8*	argp;	// pointer to arguments in old frame
	bool	panic;	// is this frame the top of a panic?
};
struct	SigTab
{
	int32	flags;
	int8	*name;
};
enum
{
	SigNotify = 1<<0,	// let signal.Notify have signal, even if from kernel
	SigKill = 1<<1,		// if signal.Notify doesn't take it, exit quietly
	SigThrow = 1<<2,	// if signal.Notify doesn't take it, exit loudly
	SigPanic = 1<<3,	// if the signal is from the kernel, panic
	SigDefault = 1<<4,	// if the signal isn't explicitly requested, don't monitor it
	SigHandling = 1<<5,	// our signal handler is registered
	SigIgnored = 1<<6,	// the signal was ignored before we registered for it
	SigGoExit = 1<<7,	// cause all runtime procs to exit (only used on Plan 9).
};

// Layout of in-memory per-function information prepared by linker
// See http://golang.org/s/go12symtab.
// Keep in sync with linker and with ../../libmach/sym.c
// and with package debug/gosym and with symtab.go in package runtime.
struct	Func
{
	uintptr	entry;	// start pc
	int32	nameoff;// function name
	
	int32	args;	// in/out args size
	int32	frame;	// legacy frame size; use pcsp if possible

	int32	pcsp;
	int32	pcfile;
	int32	pcln;
	int32	npcdata;
	int32	nfuncdata;
};

// layout of Itab known to compilers
// allocated in non-garbage-collected memory
struct	Itab
{
	InterfaceType*	inter;
	Type*	type;
	Itab*	link;
	int32	bad;
	int32	unused;
	void	(*fun[])(void);
};

#ifdef GOOS_nacl
enum {
   NaCl = 1,
};
#else
enum {
   NaCl = 0,
};
#endif

#ifdef GOOS_windows
enum {
   Windows = 1
};
#else
enum {
   Windows = 0
};
#endif
#ifdef GOOS_solaris
enum {
   Solaris = 1
};
#else
enum {
   Solaris = 0
};
#endif

// Lock-free stack node.
struct LFNode
{
	LFNode	*next;
	uintptr	pushcnt;
};

// Parallel for descriptor.
struct ParFor
{
	void (*body)(ParFor*, uint32);	// executed for each element
	uint32 done;			// number of idle threads
	uint32 nthr;			// total number of threads
	uint32 nthrmax;			// maximum number of threads
	uint32 thrseq;			// thread id sequencer
	uint32 cnt;			// iteration space [0, cnt)
	void *ctx;			// arbitrary user context
	bool wait;			// if true, wait while all threads finish processing,
					// otherwise parfor may return while other threads are still working
	ParForThread *thr;		// array of thread descriptors
	uint32 pad;			// to align ParForThread.pos for 64-bit atomic operations
	// stats
	uint64 nsteal;
	uint64 nstealcnt;
	uint64 nprocyield;
	uint64 nosyield;
	uint64 nsleep;
};

// Track memory allocated by code not written in Go during a cgo call,
// so that the garbage collector can see them.
struct CgoMal
{
	CgoMal	*next;
	void	*alloc;
};

// Holds variables parsed from GODEBUG env var.
struct DebugVars
{
	int32	allocfreetrace;
	int32	efence;
	int32	gctrace;
	int32	gcdead;
	int32	scheddetail;
	int32	schedtrace;
	int32	scavenge;
};

struct ForceGCState
{
	Mutex	lock;
	G*	g;
	uint32	idle;
};

extern bool runtime·precisestack;
extern bool runtime·copystack;

/*
 * defined macros
 *    you need super-gopher-guru privilege
 *    to add this list.
 */
#define	nelem(x)	(sizeof(x)/sizeof((x)[0]))
#define	nil		((void*)0)
#define	offsetof(s,m)	(uint32)(&(((s*)0)->m))
#define	ROUND(x, n)	(((x)+(n)-1)&~(uintptr)((n)-1)) /* all-caps to mark as macro: it evaluates n twice */

/*
 * known to compiler
 */
enum {
	Structrnd = sizeof(uintreg),
};

byte*	runtime·startup_random_data;
uint32	runtime·startup_random_data_len;

enum {
	// hashinit wants this many random bytes
	HashRandomBytes = 32
};

uint32  runtime·readgstatus(G *gp);
void    runtime·casgstatus(G*, uint32, uint32);

/*
 * deferred subroutine calls
 */
struct Defer
{
	int32	siz;
	bool	special;	// not part of defer frame
	uintptr	argp;		// where args were copied from
	uintptr	pc;
	FuncVal*	fn;
	Defer*	link;
	void*	args[1];	// padded to actual size
};

// argp used in Defer structs when there is no argp.
#define NoArgs ((uintptr)-1)

/*
 * panics
 */
struct Panic
{
	Eface	arg;		// argument to panic
	uintptr	stackbase;	// g->stackbase in panic
	Panic*	link;		// link to earlier panic
	Defer*	defer;		// current executing defer
	bool	recovered;	// whether this panic is over
	bool	aborted;	// the panic was aborted
};

typedef struct XXX XXX;

/*
 * stack traces
 */
typedef struct Stkframe Stkframe;
struct Stkframe
{
	Func*	fn;	// function being run
	uintptr	pc;	// program counter within fn
	uintptr	continpc;	// program counter where execution can continue, or 0 if not
	uintptr	lr;	// program counter at caller aka link register
	uintptr	sp;	// stack pointer at pc
	uintptr	fp;	// stack pointer at caller aka frame pointer
	uintptr	varp;	// top of local variables
	uintptr	argp;	// pointer to function arguments
	uintptr	arglen;	// number of bytes at argp
};

intgo	runtime·gentraceback(uintptr, uintptr, uintptr, G*, intgo, uintptr*, intgo, bool(**)(Stkframe*, void*), void*, bool);
void	runtime·traceback(uintptr pc, uintptr sp, uintptr lr, G* gp);
void	runtime·tracebackothers(G*);
bool	runtime·haszeroargs(uintptr pc);
bool	runtime·topofstack(Func*);
enum
{
	// The maximum number of frames we print for a traceback
	TracebackMaxFrames = 100,
};

/*
 * external data
 */
extern	String	runtime·emptystring;
extern	uintptr runtime·zerobase;
extern	G**	runtime·allg;
extern	Slice	runtime·allgs; // []*G
extern	uintptr runtime·allglen;
extern	G*	runtime·lastg;
extern	M*	runtime·allm;
extern	P*	runtime·allp[MaxGomaxprocs+1];
extern	int32	runtime·gomaxprocs;
extern	uint32	runtime·needextram;
extern	uint32	runtime·panicking;
extern	int8*	runtime·goos;
extern	int32	runtime·ncpu;
extern	bool	runtime·iscgo;
extern 	void	(*runtime·sysargs)(int32, uint8**);
extern	uintptr	runtime·maxstring;
extern	uint32	runtime·cpuid_ecx;
extern	uint32	runtime·cpuid_edx;
extern	DebugVars	runtime·debug;
extern	uintptr	runtime·maxstacksize;
extern	Note	runtime·signote;
extern	ForceGCState	runtime·forcegc;

/*
 * common functions and data
 */
int32	runtime·strcmp(byte*, byte*);
int32	runtime·strncmp(byte*, byte*, uintptr);
byte*	runtime·strstr(byte*, byte*);
intgo	runtime·findnull(byte*);
intgo	runtime·findnullw(uint16*);
void	runtime·dump(byte*, int32);
int32	runtime·runetochar(byte*, int32);
int32	runtime·charntorune(int32*, uint8*, int32);


/*
 * This macro is used when writing C functions
 * called as if they were Go functions.
 * Passed the address of a result before a return statement,
 * it makes sure the result has been flushed to memory
 * before the return.
 *
 * It is difficult to write such functions portably, because
 * of the varying requirements on the alignment of the
 * first output value. Almost all code should write such
 * functions in .goc files, where goc2c (part of cmd/dist)
 * can arrange the correct alignment for the target system.
 * Goc2c also takes care of conveying to the garbage collector
 * which parts of the argument list are inputs vs outputs.
 *
 * Therefore, do NOT use this macro if at all possible.
 */ 
#define FLUSH(x)	USED(x)

/*
 * GoOutput is a type with the same alignment requirements as the
 * initial output argument from a Go function. Only for use in cases
 * where using goc2c is not possible. See comment on FLUSH above.
 */
typedef uint64 GoOutput;

void	runtime·gogo(Gobuf*);
void	runtime·gostartcall(Gobuf*, void(*)(void), void*);
void	runtime·gostartcallfn(Gobuf*, FuncVal*);
void	runtime·gosave(Gobuf*);
void	runtime·lessstack(void);
void	runtime·goargs(void);
void	runtime·goenvs(void);
void	runtime·goenvs_unix(void);
void*	runtime·getu(void);
void	runtime·throw(int8*);
void	runtime·panicstring(int8*);
bool	runtime·canpanic(G*);
void	runtime·prints(int8*);
void	runtime·printf(int8*, ...);
void	runtime·snprintf(byte*, int32, int8*, ...);
byte*	runtime·mchr(byte*, byte, byte*);
int32	runtime·mcmp(byte*, byte*, uintptr);
void	runtime·memmove(void*, void*, uintptr);
String	runtime·catstring(String, String);
String	runtime·gostring(byte*);
String  runtime·gostringn(byte*, intgo);
Slice	runtime·gobytes(byte*, intgo);
String	runtime·gostringnocopy(byte*);
String	runtime·gostringw(uint16*);
void	runtime·initsig(void);
void	runtime·sigenable(uint32 sig);
void	runtime·sigdisable(uint32 sig);
int32	runtime·gotraceback(bool *crash);
void	runtime·goroutineheader(G*);
int32	runtime·open(int8*, int32, int32);
int32	runtime·read(int32, void*, int32);
int32	runtime·write(uintptr, void*, int32); // use uintptr to accommodate windows.
int32	runtime·close(int32);
int32	runtime·mincore(void*, uintptr, byte*);
void	runtime·jmpdefer(FuncVal*, uintptr);
void	runtime·exit1(int32);
void	runtime·ready(G*);
byte*	runtime·getenv(int8*);
int32	runtime·atoi(byte*);
void	runtime·newosproc(M *mp, void *stk);
void	runtime·mstart(void);
G*	runtime·malg(int32);
void	runtime·asminit(void);
void	runtime·mpreinit(M*);
void	runtime·minit(void);
void	runtime·unminit(void);
void	runtime·signalstack(byte*, int32);
void	runtime·symtabinit(void);
Func*	runtime·findfunc(uintptr);
int32	runtime·funcline(Func*, uintptr, String*);
int32	runtime·funcarglen(Func*, uintptr);
int32	runtime·funcspdelta(Func*, uintptr);
int8*	runtime·funcname(Func*);
int32	runtime·pcdatavalue(Func*, int32, uintptr);
void	runtime·stackinit(void);
void*	runtime·stackalloc(G*, uint32);
void	runtime·stackfree(G*, void*, Stktop*);
void	runtime·shrinkstack(G*);
MCache*	runtime·allocmcache(void);
void	runtime·freemcache(MCache*);
void	runtime·mallocinit(void);
void	runtime·gcinit(void);
void*	runtime·mallocgc(uintptr size, Type* typ, uint32 flag);
void	runtime·runpanic(Panic*);
uintptr	runtime·getcallersp(void*);
int32	runtime·mcount(void);
int32	runtime·gcount(void);
void	runtime·mcall(void(*)(G*));
void	runtime·onM(void(*)(void));
uint32	runtime·fastrand1(void);
void	runtime·rewindmorestack(Gobuf*);
int32	runtime·timediv(int64, int32, int32*);
int32	runtime·round2(int32 x); // round x up to a power of 2.

// atomic operations
bool	runtime·cas(uint32*, uint32, uint32);
bool	runtime·cas64(uint64*, uint64, uint64);
bool	runtime·casp(void**, void*, void*);
// Don't confuse with XADD x86 instruction,
// this one is actually 'addx', that is, add-and-fetch.
uint32	runtime·xadd(uint32 volatile*, int32);
uint64	runtime·xadd64(uint64 volatile*, int64);
uint32	runtime·xchg(uint32 volatile*, uint32);
uint64	runtime·xchg64(uint64 volatile*, uint64);
void*	runtime·xchgp(void* volatile*, void*);
uint32	runtime·atomicload(uint32 volatile*);
void	runtime·atomicstore(uint32 volatile*, uint32);
void	runtime·atomicstore64(uint64 volatile*, uint64);
uint64	runtime·atomicload64(uint64 volatile*);
void*	runtime·atomicloadp(void* volatile*);
void	runtime·atomicstorep(void* volatile*, void*);
void	runtime·atomicor8(byte volatile*, byte);

void	runtime·setg(G*);
void	runtime·newextram(void);
void	runtime·exit(int32);
void	runtime·breakpoint(void);
void	runtime·gosched(void);
void	runtime·gosched_m(G*);
void	runtime·schedtrace(bool);
void	runtime·park(bool(*)(G*, void*), void*, String);
void	runtime·parkunlock(Mutex*, String);
void	runtime·tsleep(int64, String);
M*	runtime·newm(void);
void	runtime·goexit(void);
void	runtime·asmcgocall(void (*fn)(void*), void*);
void	runtime·entersyscall(void);
void	runtime·entersyscallblock(void);
void	runtime·exitsyscall(void);
void	runtime·entersyscallblock_m(void);
G*	runtime·newproc1(FuncVal*, byte*, int32, int32, void*);
bool	runtime·sigsend(int32 sig);
intgo	runtime·callers(intgo, uintptr*, intgo);
intgo	runtime·gcallers(G*, intgo, uintptr*, intgo);
int64	runtime·nanotime(void);	// monotonic time
int64	runtime·unixnanotime(void); // real time, can skip
void	runtime·dopanic(int32);
void	runtime·startpanic(void);
void	runtime·freezetheworld(void);
void	runtime·unwindstack(G*, byte*);
void	runtime·sigprof(uint8 *pc, uint8 *sp, uint8 *lr, G *gp, M *mp);
void	runtime·resetcpuprofiler(int32);
void	runtime·setcpuprofilerate(int32);
void	runtime·usleep(uint32);
int64	runtime·cputicks(void);
int64	runtime·tickspersecond(void);
void	runtime·blockevent(int64, intgo);
G*	runtime·netpoll(bool);
void	runtime·netpollinit(void);
int32	runtime·netpollopen(uintptr, PollDesc*);
int32   runtime·netpollclose(uintptr);
void	runtime·netpollready(G**, PollDesc*, int32);
uintptr	runtime·netpollfd(PollDesc*);
void	runtime·netpollarm(PollDesc*, int32);
void**	runtime·netpolluser(PollDesc*);
bool	runtime·netpollclosing(PollDesc*);
void	runtime·netpolllock(PollDesc*);
void	runtime·netpollunlock(PollDesc*);
void	runtime·crash(void);
void	runtime·parsedebugvars(void);
void*	runtime·funcdata(Func*, int32);
void	runtime·setmaxthreads_m(void);
G*	runtime·timejump(void);
void	runtime·iterate_itabs(void (**callback)(Itab*));
void	runtime·iterate_finq(void (*callback)(FuncVal*, byte*, uintptr, Type*, PtrType*));

#pragma	varargck	argpos	runtime·printf	1
#pragma	varargck	type	"c"	int32
#pragma	varargck	type	"d"	int32
#pragma	varargck	type	"d"	uint32
#pragma	varargck	type	"D"	int64
#pragma	varargck	type	"D"	uint64
#pragma	varargck	type	"x"	int32
#pragma	varargck	type	"x"	uint32
#pragma	varargck	type	"X"	int64
#pragma	varargck	type	"X"	uint64
#pragma	varargck	type	"p"	void*
#pragma	varargck	type	"p"	uintptr
#pragma	varargck	type	"s"	int8*
#pragma	varargck	type	"s"	uint8*
#pragma	varargck	type	"S"	String

void	runtime·stoptheworld(void);
void	runtime·starttheworld(void);
extern uint32 runtime·worldsema;

/*
 * mutual exclusion locks.  in the uncontended case,
 * as fast as spin locks (just a few user-level instructions),
 * but on the contention path they sleep in the kernel.
 * a zeroed Mutex is unlocked (no need to initialize each lock).
 */
void	runtime·lock(Mutex*);
void	runtime·unlock(Mutex*);

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
void	runtime·noteclear(Note*);
void	runtime·notesleep(Note*);
void	runtime·notewakeup(Note*);
bool	runtime·notetsleep(Note*, int64);  // false - timeout
bool	runtime·notetsleepg(Note*, int64);  // false - timeout

/*
 * low-level synchronization for implementing the above
 */
uintptr	runtime·semacreate(void);
int32	runtime·semasleep(int64);
void	runtime·semawakeup(M*);
// or
void	runtime·futexsleep(uint32*, uint32, int64);
void	runtime·futexwakeup(uint32*, uint32);

/*
 * Mutex-free stack.
 * Initialize uint64 head to 0, compare with 0 to test for emptiness.
 * The stack does not keep pointers to nodes,
 * so they can be garbage collected if there are no other pointers to nodes.
 */
void	runtime·lfstackpush(uint64 *head, LFNode *node);
LFNode*	runtime·lfstackpop(uint64 *head);

/*
 * Parallel for over [0, n).
 * body() is executed for each iteration.
 * nthr - total number of worker threads.
 * ctx - arbitrary user context.
 * if wait=true, threads return from parfor() when all work is done;
 * otherwise, threads can return while other threads are still finishing processing.
 */
ParFor*	runtime·parforalloc(uint32 nthrmax);
void	runtime·parforsetup(ParFor *desc, uint32 nthr, uint32 n, void *ctx, bool wait, void (*body)(ParFor*, uint32));
void	runtime·parfordo(ParFor *desc);
void	runtime·parforiters(ParFor*, uintptr, uintptr*, uintptr*);

/*
 * low level C-called
 */
// for mmap, we only pass the lower 32 bits of file offset to the 
// assembly routine; the higher bits (if required), should be provided
// by the assembly routine as 0.
uint8*	runtime·mmap(byte*, uintptr, int32, int32, int32, uint32);
void	runtime·munmap(byte*, uintptr);
void	runtime·madvise(byte*, uintptr, int32);
void	runtime·memclr(byte*, uintptr);
void	runtime·setcallerpc(void*, void*);
void*	runtime·getcallerpc(void*);
void	runtime·printbool(bool);
void	runtime·printbyte(int8);
void	runtime·printfloat(float64);
void	runtime·printint(int64);
void	runtime·printiface(Iface);
void	runtime·printeface(Eface);
void	runtime·printstring(String);
void	runtime·printpc(void*);
void	runtime·printpointer(void*);
void	runtime·printuint(uint64);
void	runtime·printhex(uint64);
void	runtime·printslice(Slice);
void	runtime·printcomplex(Complex128);

/*
 * runtime go-called
 */
void	runtime·newstackcall(FuncVal*, byte*, uint32);
void	reflect·call(FuncVal*, byte*, uint32, uint32);
void	runtime·panic(Eface);
void	runtime·panicindex(void);
void	runtime·panicslice(void);
void	runtime·panicdivide(void);

/*
 * runtime c-called (but written in Go)
 */
void	runtime·printany(Eface);
void	runtime·newTypeAssertionError(String*, String*, String*, String*, Eface*);
void	runtime·newErrorString(String, Eface*);
void	runtime·newErrorCString(int8*, Eface*);
void	runtime·fadd64c(uint64, uint64, uint64*);
void	runtime·fsub64c(uint64, uint64, uint64*);
void	runtime·fmul64c(uint64, uint64, uint64*);
void	runtime·fdiv64c(uint64, uint64, uint64*);
void	runtime·fneg64c(uint64, uint64*);
void	runtime·f32to64c(uint32, uint64*);
void	runtime·f64to32c(uint64, uint32*);
void	runtime·fcmp64c(uint64, uint64, int32*, bool*);
void	runtime·fintto64c(int64, uint64*);
void	runtime·f64tointc(uint64, int64*, bool*);

/*
 * wrapped for go users
 */
float64	runtime·Inf(int32 sign);
float64	runtime·NaN(void);
float32	runtime·float32frombits(uint32 i);
uint32	runtime·float32tobits(float32 f);
float64	runtime·float64frombits(uint64 i);
uint64	runtime·float64tobits(float64 f);
float64	runtime·frexp(float64 d, int32 *ep);
bool	runtime·isInf(float64 f, int32 sign);
bool	runtime·isNaN(float64 f);
float64	runtime·ldexp(float64 d, int32 e);
float64	runtime·modf(float64 d, float64 *ip);
void	runtime·semacquire(uint32*, bool);
void	runtime·semrelease(uint32*);
int32	runtime·gomaxprocsfunc(int32 n);
void	runtime·procyield(uint32);
void	runtime·osyield(void);
void	runtime·lockOSThread(void);
void	runtime·unlockOSThread(void);
bool	runtime·lockedOSThread(void);

bool	runtime·showframe(Func*, G*);
void	runtime·printcreatedby(G*);

void	runtime·ifaceE2I(InterfaceType*, Eface, Iface*);
bool	runtime·ifaceE2I2(InterfaceType*, Eface, Iface*);
uintptr	runtime·memlimit(void);

// float.c
extern float64 runtime·nan;
extern float64 runtime·posinf;
extern float64 runtime·neginf;
extern uint64 ·nan;
extern uint64 ·posinf;
extern uint64 ·neginf;
#define ISNAN(f) ((f) != (f))

enum
{
	UseSpanType = 1,
};

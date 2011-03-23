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
#else
typedef	uint32		uintptr;
typedef int32		intptr;
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
typedef	struct	Alg		Alg;
typedef	struct	Func		Func;
typedef	struct	G		G;
typedef	struct	Gobuf		Gobuf;
typedef	struct	Lock		Lock;
typedef	struct	M		M;
typedef	struct	Mem		Mem;
typedef	union	Note		Note;
typedef	struct	Slice		Slice;
typedef	struct	Stktop		Stktop;
typedef	struct	String		String;
typedef	struct	Usema		Usema;
typedef	struct	SigTab		SigTab;
typedef	struct	MCache		MCache;
typedef	struct	Iface		Iface;
typedef	struct	Itab		Itab;
typedef	struct	Eface		Eface;
typedef	struct	Type		Type;
typedef	struct	Defer		Defer;
typedef	struct	Panic		Panic;
typedef	struct	Hmap		Hmap;
typedef	struct	Hchan		Hchan;
typedef	struct	Complex64	Complex64;
typedef	struct	Complex128	Complex128;

/*
 * per-cpu declaration.
 * "extern register" is a special storage class implemented by 6c, 8c, etc.
 * on machines with lots of registers, it allocates a register that will not be
 * used in generated code.  on the x86, it allocates a slot indexed by a
 * segment register.
 *
 * amd64: allocated downwards from R15
 * x86: allocated upwards from 0(GS)
 * arm: allocated downwards from R10
 *
 * every C file linked into a Go program must include runtime.h
 * so that the C compiler knows to avoid other uses of these registers.
 * the Go compilers know to avoid them.
 */
extern	register	G*	g;
extern	register	M*	m;

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
	Gidle,
	Grunnable,
	Grunning,
	Gsyscall,
	Gwaiting,
	Gmoribund,
	Gdead,
};
enum
{
	true	= 1,
	false	= 0,
};

/*
 * structures
 */
struct	Lock
{
	uint32	key;
#ifdef __WINDOWS__
	void*	event;
#else
	uint32	sema;	// for OS X
#endif
};
struct	Usema
{
	uint32	u;
	uint32	k;
};
union	Note
{
	struct {	// Linux
		Lock	lock;
	};
	struct {	// OS X
		int32	wakeup;
		Usema	sema;
	};
};
struct String
{
	byte*	str;
	int32	len;
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
	uint32	len;		// number of elements
	uint32	cap;		// allocated number of elements
};
struct	Gobuf
{
	// The offsets of these fields are known to (hard-coded in) libmach.
	byte*	sp;
	byte*	pc;
	G*	g;
};
struct	G
{
	byte*	stackguard;	// cannot move - also known to linker, libmach, libcgo
	byte*	stackbase;	// cannot move - also known to libmach, libcgo
	Defer*	defer;
	Panic*	panic;
	Gobuf	sched;
	byte*	stack0;
	byte*	entry;		// initial function
	G*	alllink;	// on allg
	void*	param;		// passed parameter on wakeup
	int16	status;
	int32	goid;
	uint32	selgen;		// valid sudog pointer
	G*	schedlink;
	bool	readyonstop;
	bool	ispanic;
	M*	m;		// for debuggers, but offset not hard-coded
	M*	lockedm;
	M*	idlem;
	int32	sig;
	uintptr	sigcode0;
	uintptr	sigcode1;
	uintptr	sigpc;
	uintptr	gopc;	// pc of go statement that created this goroutine
};
struct	M
{
	// The offsets of these fields are known to (hard-coded in) libmach.
	G*	g0;		// goroutine with scheduling stack
	void	(*morepc)(void);
	void*	moreargp;	// argument pointer for more stack
	Gobuf	morebuf;	// gobuf arg to morestack

	// Fields not known to debuggers.
	uint32	moreframesize;	// size arguments to morestack
	uint32	moreargsize;
	uintptr	cret;		// return value from C
	uint64	procid;		// for debuggers, but offset not hard-coded
	G*	gsignal;	// signal-handling G
	uint32	tls[8];		// thread-local storage (for 386 extern register)
	G*	curg;		// current running goroutine
	int32	id;
	int32	mallocing;
	int32	gcing;
	int32	locks;
	int32	nomemprof;
	int32	waitnextg;
	int32	dying;
	int32	profilehz;
	Note	havenextg;
	G*	nextg;
	M*	alllink;	// on allm
	M*	schedlink;
	uint32	machport;	// Return address for Mach IPC (OS X)
	MCache	*mcache;
	G*	lockedg;
	G*	idleg;
	uint32	freglo[16];	// D[i] lsb and F[i]
	uint32	freghi[16];	// D[i] msb and F[i+16]
	uint32	fflag;		// floating point compare flags
#ifdef __WINDOWS__
	void*	sehframe;
#endif
};
struct	Stktop
{
	// The offsets of these fields are known to (hard-coded in) libmach.
	uint8*	stackguard;
	uint8*	stackbase;
	Gobuf	gobuf;
	uint32	argsize;

	uint8*	argp;	// pointer to arguments in old frame
	uintptr	free;	// if free>0, call stackfree using free as size
	bool	panic;	// is this frame the top of a panic?
};
struct	Alg
{
	uintptr	(*hash)(uint32, void*);
	uint32	(*equal)(uint32, void*, void*);
	void	(*print)(uint32, void*);
	void	(*copy)(uint32, void*, void*);
};
struct	SigTab
{
	int32	flags;
	int8	*name;
};
enum
{
	SigCatch = 1<<0,
	SigIgnore = 1<<1,
	SigRestart = 1<<2,
	SigQueue = 1<<3,
	SigPanic = 1<<4,
};

// NOTE(rsc): keep in sync with extern.go:/type.Func.
// Eventually, the loaded symbol table should be closer to this form.
struct	Func
{
	String	name;
	String	type;	// go type string
	String	src;	// src file name
	Slice	pcln;	// pc/ln tab for this func
	uintptr	entry;	// entry pc
	uintptr	pc0;	// starting pc, ln for table
	int32	ln0;
	int32	frame;	// stack frame size
	int32	args;	// number of 32-bit in/out args
	int32	locals;	// number of 32-bit locals
};

#ifdef __WINDOWS__
enum {
   Windows = 1
};
#else
enum {
   Windows = 0
};
#endif

/*
 * defined macros
 *    you need super-goru privilege
 *    to add this list.
 */
#define	nelem(x)	(sizeof(x)/sizeof((x)[0]))
#define	nil		((void*)0)
#define	offsetof(s,m)	(uint32)(&(((s*)0)->m))

/*
 * known to compiler
 */
enum
{
	AMEM,
	ANOEQ,
	ASTRING,
	AINTER,
	ANILINTER,
	AMEMWORD,
	Amax
};


enum {
	Structrnd = sizeof(uintptr)
};

/*
 * deferred subroutine calls
 */
struct Defer
{
	int32	siz;
	byte*	argp;  // where args were copied from
	byte*	pc;
	byte*	fn;
	Defer*	link;
	byte	args[8];	// padded to actual size
};

/*
 * panics
 */
struct Panic
{
	Eface	arg;		// argument to panic
	byte*	stackbase;	// g->stackbase in panic
	Panic*	link;		// link to earlier panic
	bool	recovered;	// whether this panic is over
};

/*
 * external data
 */
extern	Alg	runtime·algarray[Amax];
extern	String	runtime·emptystring;
G*	runtime·allg;
M*	runtime·allm;
int32	runtime·goidgen;
extern	int32	runtime·gomaxprocs;
extern	uint32	runtime·panicking;
extern	int32	runtime·gcwaiting;		// gc is waiting to run
int8*	runtime·goos;
extern	bool	runtime·iscgo;

/*
 * common functions and data
 */
int32	runtime·strcmp(byte*, byte*);
int32	runtime·findnull(byte*);
int32	runtime·findnullw(uint16*);
void	runtime·dump(byte*, int32);
int32	runtime·runetochar(byte*, int32);
int32	runtime·charntorune(int32*, uint8*, int32);

/*
 * very low level c-called
 */
#define FLUSH(x)	USED(x)

void	runtime·gogo(Gobuf*, uintptr);
void	runtime·gogocall(Gobuf*, void(*)(void));
void	runtime·gosave(Gobuf*);
void	runtime·lessstack(void);
void	runtime·goargs(void);
void	runtime·goenvs(void);
void	runtime·goenvs_unix(void);
void*	runtime·getu(void);
void	runtime·throw(int8*);
void	runtime·panicstring(int8*);
uint32	runtime·rnd(uint32, uint32);
void	runtime·prints(int8*);
void	runtime·printf(int8*, ...);
byte*	runtime·mchr(byte*, byte, byte*);
void	runtime·mcpy(byte*, byte*, uint32);
int32	runtime·mcmp(byte*, byte*, uint32);
void	runtime·memmove(void*, void*, uint32);
void*	runtime·mal(uintptr);
String	runtime·catstring(String, String);
String	runtime·gostring(byte*);
String  runtime·gostringn(byte*, int32);
String	runtime·gostringnocopy(byte*);
String	runtime·gostringw(uint16*);
void	runtime·initsig(int32);
int32	runtime·gotraceback(void);
void	runtime·traceback(uint8 *pc, uint8 *sp, uint8 *lr, G* gp);
void	runtime·tracebackothers(G*);
int32	runtime·write(int32, void*, int32);
bool	runtime·cas(uint32*, uint32, uint32);
bool	runtime·casp(void**, void*, void*);
uint32	runtime·xadd(uint32 volatile*, int32);
void	runtime·jmpdefer(byte*, void*);
void	runtime·exit1(int32);
void	runtime·ready(G*);
byte*	runtime·getenv(int8*);
int32	runtime·atoi(byte*);
void	runtime·newosproc(M *m, G *g, void *stk, void (*fn)(void));
void	runtime·signalstack(byte*, int32);
G*	runtime·malg(int32);
void	runtime·minit(void);
Func*	runtime·findfunc(uintptr);
int32	runtime·funcline(Func*, uintptr);
void*	runtime·stackalloc(uint32);
void	runtime·stackfree(void*, uintptr);
MCache*	runtime·allocmcache(void);
void	runtime·mallocinit(void);
bool	runtime·ifaceeq_c(Iface, Iface);
bool	runtime·efaceeq_c(Eface, Eface);
uintptr	runtime·ifacehash(Iface);
uintptr	runtime·efacehash(Eface);
uintptr	runtime·nohash(uint32, void*);
uint32	runtime·noequal(uint32, void*, void*);
void*	runtime·malloc(uintptr size);
void	runtime·free(void *v);
void	runtime·addfinalizer(void*, void(*fn)(void*), int32);
void	runtime·walkfintab(void (*fn)(void*));
void	runtime·runpanic(Panic*);
void*	runtime·getcallersp(void*);
int32	runtime·mcount(void);
void	runtime·mcall(void(*)(G*));

void	runtime·exit(int32);
void	runtime·breakpoint(void);
void	runtime·gosched(void);
void	runtime·goexit(void);
void	runtime·asmcgocall(void (*fn)(void*), void*);
void	runtime·entersyscall(void);
void	runtime·exitsyscall(void);
G*	runtime·newproc1(byte*, byte*, int32, int32, void*);
void	runtime·siginit(void);
bool	runtime·sigsend(int32 sig);
void	runtime·gettime(int64*, int32*);
int32	runtime·callers(int32, uintptr*, int32);
int32	runtime·gentraceback(byte*, byte*, byte*, G*, int32, uintptr*, int32);
int64	runtime·nanotime(void);
void	runtime·dopanic(int32);
void	runtime·startpanic(void);
void	runtime·sigprof(uint8 *pc, uint8 *sp, uint8 *lr, G *gp);
void	runtime·resetcpuprofiler(int32);
void	runtime·setcpuprofilerate(void(*)(uintptr*, int32), int32);

#pragma	varargck	argpos	runtime·printf	1
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

// TODO(rsc): Remove. These are only temporary,
// for the mark and sweep collector.
void	runtime·stoptheworld(void);
void	runtime·starttheworld(void);

/*
 * mutual exclusion locks.  in the uncontended case,
 * as fast as spin locks (just a few user-level instructions),
 * but on the contention path they sleep in the kernel.
 * a zeroed Lock is unlocked (no need to initialize each lock).
 */
void	runtime·lock(Lock*);
void	runtime·unlock(Lock*);
void	runtime·destroylock(Lock*);

/*
 * sleep and wakeup on one-time events.
 * before any calls to notesleep or notewakeup,
 * must call noteclear to initialize the Note.
 * then, any number of threads can call notesleep
 * and exactly one thread can call notewakeup (once).
 * once notewakeup has been called, all the notesleeps
 * will return.  future notesleeps will return immediately.
 */
void	runtime·noteclear(Note*);
void	runtime·notesleep(Note*);
void	runtime·notewakeup(Note*);

/*
 * This is consistent across Linux and BSD.
 * If a new OS is added that is different, move this to
 * $GOOS/$GOARCH/defs.h.
 */
#define EACCES		13

/*
 * low level C-called
 */
uint8*	runtime·mmap(byte*, uintptr, int32, int32, int32, uint32);
void	runtime·munmap(uint8*, uintptr);
void	runtime·memclr(byte*, uintptr);
void	runtime·setcallerpc(void*, void*);
void*	runtime·getcallerpc(void*);

/*
 * runtime go-called
 */
void	runtime·printbool(bool);
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
void	reflect·call(byte*, byte*, uint32);
void	runtime·panic(Eface);
void	runtime·panicindex(void);
void	runtime·panicslice(void);

/*
 * runtime c-called (but written in Go)
 */
void	runtime·newError(String, Eface*);
void	runtime·printany(Eface);
void	runtime·newTypeAssertionError(Type*, Type*, Type*, String*, String*, String*, String*, Eface*);
void	runtime·newErrorString(String, Eface*);
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
void	runtime·semacquire(uint32*);
void	runtime·semrelease(uint32*);
String	runtime·signame(int32 sig);
int32	runtime·gomaxprocsfunc(int32 n);

void	runtime·mapassign(Hmap*, byte*, byte*);
void	runtime·mapaccess(Hmap*, byte*, byte*, bool*);
struct hash_iter*	runtime·newmapiterinit(Hmap*);
void	runtime·mapiternext(struct hash_iter*);
bool	runtime·mapiterkey(struct hash_iter*, void*);
void	runtime·mapiterkeyvalue(struct hash_iter*, void*, void*);
Hmap*	runtime·makemap_c(Type*, Type*, int64);

Hchan*	runtime·makechan_c(Type*, int64);
void	runtime·chansend(Hchan*, void*, bool*);
void	runtime·chanrecv(Hchan*, void*, bool*, bool*);
void	runtime·chanclose(Hchan*);
int32	runtime·chanlen(Hchan*);
int32	runtime·chancap(Hchan*);

void	runtime·ifaceE2I(struct InterfaceType*, Eface, Iface*);

enum
{
	// StackSystem is a number of additional bytes to add
	// to each stack below the usual guard area for OS-specific
	// purposes like signal handling.
	// TODO(rsc): This is only for Windows.  Can't Windows use
	// a separate exception stack like every other operating system?
#ifdef __WINDOWS__
	StackSystem = 2048,
#else
	StackSystem = 0,
#endif
};


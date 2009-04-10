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
#else
typedef	uint32		uintptr;
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
typedef	struct	Array		Array;
typedef	struct	Func		Func;
typedef	struct	G		G;
typedef	struct	Gobuf		Gobuf;
typedef	struct	Lock		Lock;
typedef	struct	M		M;
typedef	struct	Mem		Mem;
typedef	union	Note		Note;
typedef	struct	Stktop		Stktop;
typedef	struct	String		String;
typedef	struct	Usema		Usema;
typedef	struct	SigTab		SigTab;
typedef	struct	MCache		MCache;
typedef	struct	Iface		Iface;
typedef	struct	Itype		Itype;
typedef	struct	Defer		Defer;

/*
 * per cpu declaration
 */
extern	register	G*	g;	// R15
extern	register	M*	m;	// R14

/*
 * defined constants
 */
enum
{
	// G status
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
	uint32	sema;	// for OS X
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
	Itype*	type;
	void*	data;
};

struct	Array
{				// must not move anything
	byte*	array;		// actual data
	uint32	nel;		// number of elements
	uint32	cap;		// allocated number of elements
};
struct	Gobuf
{
	byte*	SP;
	byte*	PC;
};
struct	G
{
	byte*	stackguard;	// must not move
	byte*	stackbase;	// must not move
	Defer*	defer;		// must not move
	byte*	stack0;		// first stack segment
	Gobuf	sched;
	G*	alllink;	// on allg
	void*	param;		// passed parameter on wakeup
	int16	status;
	int32	goid;
	int32	selgen;		// valid sudog pointer
	G*	schedlink;
	bool	readyonstop;
	M*	m;		// for debuggers
};
struct	Mem
{
	uint8*	hunk;
	uint32	nhunk;
	uint64	nmmap;
	uint64	nmal;
};
struct	M
{
	G*	g0;		// g0 w interrupt stack - must not move
	uint64	morearg;	// arg to morestack - must not move
	uint64	cret;		// return value from C - must not move
	uint64	procid;		// for debuggers - must not move
	G*	gsignal;	// signal-handling G - must not move
	G*	curg;		// current running goroutine - must not move
	G*	lastg;		// last running goroutine - to emulate fifo
	Gobuf	sched;
	Gobuf	morestack;
	byte*	moresp;
	int32	siz1;
	int32	siz2;
	int32	id;
	int32	mallocing;
	int32	locks;
	Note	havenextg;
	G*	nextg;
	M*	schedlink;
	Mem	mem;
	uint32	machport;	// Return address for Mach IPC (OS X)
	MCache	*mcache;
};
struct	Stktop
{
	uint8*	oldbase;
	uint8*	oldsp;
	uint64	magic;
	uint8*	oldguard;
};
struct	Alg
{
	uint64	(*hash)(uint32, void*);
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
};

// (will be) shared with go; edit ../cmd/6g/sys.go too.
// should move out of sys.go eventually.
// also eventually, the loaded symbol table should
// be closer to this form.
struct	Func
{
	String	name;
	String	type;	// go type string
	String	src;	// src file name
	uint64	entry;	// entry pc
	int64	frame;	// stack frame size
	Array	pcln;	// pc/ln tab for this func
	int64	pc0;	// starting pc, ln for table
	int32	ln0;
	int32	args;	// number of 32-bit in/out args
	int32	locals;	// number of 32-bit locals
};

/*
 * defined macros
 *    you need super-goru privilege
 *    to add this list.
 */
#define	nelem(x)	(sizeof(x)/sizeof((x)[0]))
#define	nil		((void*)0)

/*
 * known to compiler
 */
enum
{
	AMEM,
	ANOEQ,
	ASTRING,
	AINTER,
	AFAKE,
	Amax
};

/*
 * deferred subroutine calls
 */
struct Defer
{
	int32	siz;
	byte*	sp;
	byte*	fn;
	Defer*	link;
	byte	args[8];	// padded to actual size
};

/*
 * external data
 */
extern	Alg	algarray[Amax];
extern	String	emptystring;
G*	allg;
int32	goidgen;
extern	int32	gomaxprocs;
extern	int32	panicking;
extern	int32	maxround;

/*
 * common functions and data
 */
int32	strcmp(byte*, byte*);
int32	findnull(byte*);
void	dump(byte*, int32);
int32	runetochar(byte*, int32);

/*
 * very low level c-called
 */
int32	gogo(Gobuf*);
int32	gosave(Gobuf*);
int32	gogoret(Gobuf*, uint64);
void	retfromnewstack(void);
void	goargs(void);
void	setspgoto(byte*, void(*)(void), void(*)(void));
void	FLUSH(void*);
void*	getu(void);
void	throw(int8*);
uint32	rnd(uint32, uint32);
void	prints(int8*);
void	printf(int8*, ...);
byte*	mchr(byte*, byte, byte*);
void	mcpy(byte*, byte*, uint32);
int32	mcmp(byte*, byte*, uint32);
void	mmov(byte*, byte*, uint32);
void*	mal(uint32);
uint32	cmpstring(String, String);
String	gostring(byte*);
void	initsig(void);
int32	gotraceback(void);
void	traceback(uint8 *pc, uint8 *sp, G* gp);
void	tracebackothers(G*);
int32	open(byte*, int32, ...);
int32	read(int32, void*, int32);
int32	write(int32, void*, int32);
void	close(int32);
int32	fstat(int32, void*);
bool	cas(uint32*, uint32, uint32);
void	jmpdefer(byte*);
void	exit1(int32);
void	ready(G*);
byte*	getenv(int8*);
int32	atoi(byte*);
void	newosproc(M *m, G *g, void *stk, void (*fn)(void));
void	sigaltstack(void*, void*);
void	signalstack(byte*, int32);
G*	malg(int32);
void	minit(void);
Func*	findfunc(uintptr);
int32	funcline(Func*, uint64);
void*	stackalloc(uint32);
void	stackfree(void*);
MCache*	allocmcache(void);
void	mallocinit(void);
bool	ifaceeq(Iface, Iface);
uint64	ifacehash(Iface);
uint64	nohash(uint32, void*);
uint32	noequal(uint32, void*, void*);
void*	malloc(uintptr size);
void*	mallocgc(uintptr size);
void	free(void *v);

#pragma	varargck	argpos	printf	1

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
void	stoptheworld(void);
void	starttheworld(void);

/*
 * mutual exclusion locks.  in the uncontended case,
 * as fast as spin locks (just a few user-level instructions),
 * but on the contention path they sleep in the kernel.
 * a zeroed Lock is unlocked (no need to initialize each lock).
 */
void	lock(Lock*);
void	unlock(Lock*);

/*
 * sleep and wakeup on one-time events.
 * before any calls to notesleep or notewakeup,
 * must call noteclear to initialize the Note.
 * then, any number of threads can call notesleep
 * and exactly one thread can call notewakeup (once).
 * once notewakeup has been called, all the notesleeps
 * will return.  future notesleeps will return immediately.
 */
void	noteclear(Note*);
void	notesleep(Note*);
void	notewakeup(Note*);

/*
 * Redefine methods for the benefit of gcc, which does not support
 * UTF-8 characters in identifiers.
 */
#ifndef __GNUC__
#define sys_Exit sys·Exit
#define sys_Gosched sys·Gosched
#define sys_memclr sys·memclr
#define sys_write sys·write
#define sys_Breakpoint sys·Breakpoint
#define sys_catstring sys·catstring
#define sys_cmpstring sys·cmpstring
#define sys_getcallerpc sys·getcallerpc
#define sys_Goexit sys·Goexit
#define sys_indexstring sys·indexstring
#define sys_intstring sys·intstring
#define sys_mal sys·mal
#define sys_mmap sys·mmap
#define sys_printarray sys·printarray
#define sys_printbool sys·printbool
#define sys_printfloat sys·printfloat
#define sys_printhex sys·printhex
#define sys_printint sys·printint
#define sys_printinter sys·printinter
#define sys_printpc sys·printpc
#define sys_printpointer sys·printpointer
#define sys_printstring sys·printstring
#define sys_printuint sys·printuint
#define sys_setcallerpc sys·setcallerpc
#define sys_slicestring sys·slicestring
#endif

/*
 * low level go-called
 */
void	sys_Goexit(void);
void	sys_Gosched(void);
void	sys_Exit(int32);
void	sys_write(int32, void*, int32);
void	sys_Breakpoint(void);
uint8*	sys_mmap(byte*, uint32, int32, int32, int32, uint32);
void	sys_memclr(byte*, uint32);
void	sys_setcallerpc(void*, void*);
void*	sys_getcallerpc(void*);

/*
 * runtime go-called
 */
void	sys_printbool(bool);
void	sys_printfloat(float64);
void	sys_printint(int64);
void	sys_printinter(Iface);
void	sys_printstring(String);
void	sys_printpc(void*);
void	sys_printpointer(void*);
void	sys_printuint(uint64);
void	sys_printhex(uint64);
void	sys_printarray(Array);
void	sys_catstring(String, String, String);
void	sys_cmpstring(String, String, int32);
void	sys_slicestring(String, int32, int32, String);
void	sys_indexstring(String, int32, byte);
void	sys_intstring(int64, String);

/*
 * wrapped for go users
 */
float64	Inf(int32 sign);
float64	NaN(void);
float32	float32frombits(uint32 i);
uint32	float32tobits(float32 f);
float64	float64frombits(uint64 i);
uint64	float64tobits(float64 f);
float64	frexp(float64 d, int32 *ep);
bool	isInf(float64 f, int32 sign);
bool	isNaN(float64 f);
float64	ldexp(float64 d, int32 e);
float64	modf(float64 d, float64 *ip);
void	semacquire(uint32*);
void	semrelease(uint32*);

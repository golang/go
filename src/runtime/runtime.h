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
typedef	uint64		uintptr;

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
typedef	struct	String		*string;
typedef	struct	Usema		Usema;
typedef	struct	SigTab	SigTab;

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
enum
{
	SmallFreeClasses = 168,	// number of small free lists in malloc
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
	int32	len;
	byte	str[1];
};

struct	Array
{				// must not move anything
	byte*	array;		// actual data
	uint32	nel;		// number of elements
	uint32	cap;		// allocated number of elements
	byte	b[8];		// actual array - may not be contig
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
	byte*	stack0;		// first stack segment
	Gobuf	sched;
	G*	alllink;	// on allg
	void*	param;		// passed parameter on wakeup
	int16	status;
	int32	goid;
	int32	selgen;		// valid sudog pointer
	G*	schedlink;
	bool		readyonstop;
	M*	m;	// for debuggers
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
	uint64	procid;	// for debuggers - must not move
	G*	gsignal;		// signal-handling G - must not move
	G*	curg;		// current running goroutine
	G*	lastg;		// last running goroutine - to emulate fifo
	Gobuf	sched;
	Gobuf	morestack;
	byte*	moresp;
	int32	siz1;
	int32	siz2;
	int32	id;
	Note	havenextg;
	G*	nextg;
	M*	schedlink;
	Mem	mem;
	uint32	machport;	// Return address for Mach IPC (OS X)
	void*	freelist[SmallFreeClasses];
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
	string	name;
	string	type;	// go type string
	string	src;	// src file name
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
 * external data
 */
extern	Alg	algarray[4];
extern	string	emptystring;
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
int32	chartorune(uint32*, byte*);

/*
 * very low level c-called
 */
int32	gogo(Gobuf*);
int32	gosave(Gobuf*);
int32	gogoret(Gobuf*, uint64);
void	retfromnewstack(void);
void	setspgoto(byte*, void(*)(void), void(*)(void));
void	FLUSH(void*);
void*	getu(void);
void	throw(int8*);
uint32	rnd(uint32, uint32);
void	prints(int8*);
void	printf(int8*, ...);
byte*	mchr(byte*, byte, byte*);
void	mcpy(byte*, byte*, uint32);
void	mmov(byte*, byte*, uint32);
void*	mal(uint32);
uint32	cmpstring(string, string);
string	gostring(byte*);
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
void	exit1(int32);
void	ready(G*);
byte*	getenv(int8*);
int32	atoi(byte*);
void	newosproc(M *m, G *g, void *stk, void (*fn)(void));
void	sigaltstack(void*, void*);
void	signalstack(byte*, int32);
G*	malg(int32);
void	minit(void);
Func*	findfunc(uint64);
int32	funcline(Func*, uint64);
void*	stackalloc(uint32);
void	stackfree(void*);

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
 * low level go -called
 */
void	sys·goexit(void);
void	sys·gosched(void);
void	sys·exit(int32);
void	sys·write(int32, void*, int32);
void	sys·breakpoint(void);
uint8*	sys·mmap(byte*, uint32, int32, int32, int32, uint32);
void	sys·memclr(byte*, uint32);
void	sys·setcallerpc(void*, void*);
void*	sys·getcallerpc(void*);

/*
 * runtime go-called
 */
void	sys·printbool(bool);
void	sys·printfloat(float64);
void	sys·printint(int64);
void	sys·printstring(string);
void	sys·printpc(void*);
void	sys·printpointer(void*);
void	sys·printuint(uint64);
void	sys·printhex(uint64);
void	sys·catstring(string, string, string);
void	sys·cmpstring(string, string, int32);
void	sys·slicestring(string, int32, int32, string);
void	sys·indexstring(string, int32, byte);
void	sys·intstring(int64, string);
bool	isInf(float64, int32);
bool	isNaN(float64);

/*
 * User go-called
 */
void	sys·readfile(string, string, bool);
void	sys·bytestorune(byte*, int32, int32, int32, int32);
void	sys·stringtorune(string, int32, int32, int32);
void	sys·semacquire(uint32*);
void	sys·semrelease(uint32*);

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Plan 9-specific system calls
int32	runtime·pread(int32 fd, void *buf, int32 nbytes, int64 offset);
int32	runtime·pwrite(int32 fd, void *buf, int32 nbytes, int64 offset);
int64	runtime·seek(int32 fd, int64 offset, int32 whence);
void	runtime·exits(int8* msg);
intptr	runtime·brk_(void*);
int32	runtime·sleep(int32 ms);
int32	runtime·rfork(int32 flags);
int32	runtime·plan9_semacquire(uint32 *addr, int32 block);
int32	runtime·plan9_tsemacquire(uint32 *addr, int32 ms);
int32 	runtime·plan9_semrelease(uint32 *addr, int32 count);
int32	runtime·notify(void (*fn)(void*, int8*));
int32	runtime·noted(int32);
int64	runtime·nsec(int64*);
void	runtime·sigtramp(void*, int8*);
void	runtime·sigpanic(void);
void	runtime·goexitsall(int8*);
void	runtime·setfpmasks(void);
void	runtime·tstart_plan9(M *newm);

/* open */
enum
{
	OREAD	= 0,
	OWRITE	= 1,
	ORDWR	= 2,
	OEXEC	= 3,
	OTRUNC	= 16,
	OCEXEC	= 32,
	ORCLOSE	= 64,
	OEXCL	= 0x1000
};

/* rfork */
enum
{
	RFNAMEG         = (1<<0),
	RFENVG          = (1<<1),
	RFFDG           = (1<<2),
	RFNOTEG         = (1<<3),
	RFPROC          = (1<<4),
	RFMEM           = (1<<5),
	RFNOWAIT        = (1<<6),
	RFCNAMEG        = (1<<10),
	RFCENVG         = (1<<11),
	RFCFDG          = (1<<12),
	RFREND          = (1<<13),
	RFNOMNT         = (1<<14)
};

/* notify */
enum
{
	NCONT	= 0,
	NDFLT	= 1
};

typedef struct Tos Tos;
typedef intptr _Plink;

struct Tos {
	struct TosProf			/* Per process profiling */
	{
		_Plink	*pp;	/* known to be 0(ptr) */
		_Plink	*next;	/* known to be 4(ptr) */
		_Plink	*last;
		_Plink	*first;
		uint32	pid;
		uint32	what;
	} prof;
	uint64	cyclefreq;	/* cycle clock frequency if there is one, 0 otherwise */
	int64	kcycles;	/* cycles spent in kernel */
	int64	pcycles;	/* cycles spent in process (kernel + user) */
	uint32	pid;		/* might as well put the pid here */
	uint32	clock;
	/* top of stack is here */
};

enum {
	NSIG = 14, /* number of signals in runtime·SigTab array */
	ERRMAX = 128, /* max length of note string */

	/* Notes in runtime·sigtab that are handled by runtime·sigpanic. */
	SIGRFAULT = 2,
	SIGWFAULT = 3,
	SIGINTDIV = 4,
	SIGFLOAT = 5,
	SIGTRAP = 6,
};

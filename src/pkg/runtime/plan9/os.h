// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

extern int32 runtime·write(int32 fd, void* buffer, int32 nbytes);
extern void runtime·exits(int8* msg);
extern int32 runtime·brk_(void*);

/* open */
enum
{
	OREAD			= 0,
	OWRITE			= 1,
	ORDWR			= 2
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

typedef struct Tos Tos;
typedef intptr Plink;

struct Tos {
	struct			/* Per process profiling */
	{
		Plink	*pp;	/* known to be 0(ptr) */
		Plink	*next;	/* known to be 4(ptr) */
		Plink	*last;
		Plink	*first;
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

extern int32 runtime·rfork(int32 flags, void *stk, M *m, G *g, void (*fn)(void));
extern int32 runtime·plan9_semacquire(uint32 *addr, int32 block);
extern int32 runtime·plan9_semrelease(uint32 *addr, int32 count);

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define	MAXALIGN	8

typedef	struct	WaitQ	WaitQ;
typedef	struct	SudoG	SudoG;
typedef	struct	Select	Select;
typedef	struct	Scase	Scase;

// Known to compiler.
// Changes here must also be made in src/cmd/gc/select.c's selecttype.
struct	SudoG
{
	G*	g;
	uint32*	selectdone;
	SudoG*	link;
	byte*	elem;		// data element
	int64	releasetime;
};

struct	WaitQ
{
	SudoG*	first;
	SudoG*	last;
};

struct	Hchan
{
	uintgo	qcount;			// total data in the q
	uintgo	dataqsiz;		// size of the circular q
	byte*	buf;
	uint16	elemsize;
	bool	closed;
	Type*	elemtype;		// element type
	uintgo	sendx;			// send index
	uintgo	recvx;			// receive index
	WaitQ	recvq;			// list of recv waiters
	WaitQ	sendq;			// list of send waiters
	Lock	lock;
};

// Buffer follows Hchan immediately in memory.
// chanbuf(c, i) is pointer to the i'th slot in the buffer.
#define chanbuf(c, i) ((byte*)((c)->buf)+(uintptr)(c)->elemsize*(i))

enum
{
	debug = 0,

	// Scase.kind
	CaseRecv,
	CaseSend,
	CaseDefault,
};

// Known to compiler.
// Changes here must also be made in src/cmd/gc/select.c's selecttype.
struct	Scase
{
	SudoG	sg;			// must be first member (cast to Scase)
	Hchan*	chan;			// chan
	byte*	pc;			// return pc
	uint16	kind;
	uint16	so;			// vararg of selected bool
	bool*	receivedp;		// pointer to received bool (recv2)
};

// Known to compiler.
// Changes here must also be made in src/cmd/gc/select.c's selecttype.
struct	Select
{
	uint16	tcase;			// total count of scase[]
	uint16	ncase;			// currently filled scase[]
	uint16*	pollorder;		// case poll order
	Hchan**	lockorder;		// channel lock order
	Scase	scase[1];		// one per case (in order of appearance)
};

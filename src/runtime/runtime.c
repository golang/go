// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

int32	debug	= 0;

/*BUG: move traceback code to architecture-dependent runtime */
void
sys·panicl(int32 lno)
{
	uint8 *sp;

	prints("\npanic on line ");
	sys·printint(lno);
	prints(" ");
	sys·printpc(&lno);
	prints("\n");
	sp = (uint8*)&lno;
	traceback(sys·getcallerpc(&lno), sp, getu());
	sys·breakpoint();
	sys·exit(2);
}

static	uint8*	hunk;
static	uint32	nhunk;
static	uint64	nmmap;
static	uint64	nmal;
enum
{
	NHUNK		= 20<<20,

	PROT_NONE	= 0x00,
	PROT_READ	= 0x01,
	PROT_WRITE	= 0x02,
	PROT_EXEC	= 0x04,

	MAP_FILE	= 0x0000,
	MAP_SHARED	= 0x0001,
	MAP_PRIVATE	= 0x0002,
	MAP_FIXED	= 0x0010,
	MAP_ANON	= 0x1000,
};

void
throw(int8 *s)
{
	prints("throw: ");
	prints(s);
	prints("\n");
	*(int32*)0 = 0;
	sys·exit(1);
}

void
mcpy(byte *t, byte *f, uint32 n)
{
	while(n > 0) {
		*t = *f;
		t++;
		f++;
		n--;
	}
}

static byte*
brk(uint32 n)
{
	byte* v;

	v = sys·mmap(nil, n, PROT_READ|PROT_WRITE, MAP_ANON|MAP_PRIVATE, 0, 0);
	sys·memclr(v, n);
	nmmap += n;
	return v;
}

void*
mal(uint32 n)
{
	byte* v;

	// round to keep everything 64-bit aligned
	n = (n+7) & ~7;
	nmal += n;

	// do we have enough in contiguous hunk
	if(n > nhunk) {

		// if it is big allocate it separately
		if(n > NHUNK)
			return brk(n);

		// allocate a new contiguous hunk
		hunk = brk(NHUNK);
		nhunk = NHUNK;
	}

	// allocate from the contiguous hunk
	v = hunk;
	hunk += n;
	nhunk -= n;
	return v;
}

void
sys·mal(uint32 n, uint8 *ret)
{
	ret = mal(n);
	FLUSH(&ret);
}

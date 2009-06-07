// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

#pragma textflag 7
// func closure(siz int32,
//	fn func(arg0, arg1, arg2 *ptr, callerpc uintptr, xxx) yyy,
//	arg0, arg1, arg2 *ptr) (func(xxx) yyy)
void
sys·closure(int32 siz, byte *fn, byte *arg0)
{
	byte *p, *q, **ret;
	int32 i, n;
	int64 pcrel;

	if(siz < 0 || siz%8 != 0)
		throw("bad closure size");

	ret = (byte**)((byte*)&arg0 + siz);

	if(siz > 100) {
		// TODO(rsc): implement stack growth preamble?
		throw("closure too big");
	}

	// compute size of new fn.
	// must match code laid out below.
	n = 7+10+3;	// SUBQ MOVQ MOVQ
	if(siz <= 4*8)
		n += 2*siz/8;	// MOVSQ MOVSQ...
	else
		n += 7+3;	// MOVQ REP MOVSQ
	n += 12;	// CALL worst case; sometimes only 5
	n += 7+1;	// ADDQ RET

	// store args aligned after code, so gc can find them.
	n += siz;
	if(n%8)
		n += 8 - n%8;

	p = mal(n);
	*ret = p;
	q = p + n - siz;
	mcpy(q, (byte*)&arg0, siz);

	// SUBQ $siz, SP
	*p++ = 0x48;
	*p++ = 0x81;
	*p++ = 0xec;
	*(uint32*)p = siz;
	p += 4;

	// MOVQ $q, SI
	*p++ = 0x48;
	*p++ = 0xbe;
	*(byte**)p = q;
	p += 8;

	// MOVQ SP, DI
	*p++ = 0x48;
	*p++ = 0x89;
	*p++ = 0xe7;

	if(siz <= 4*8) {
		for(i=0; i<siz; i+=8) {
			// MOVSQ
			*p++ = 0x48;
			*p++ = 0xa5;
		}
	} else {
		// MOVQ $(siz/8), CX  [32-bit immediate siz/8]
		*p++ = 0x48;
		*p++ = 0xc7;
		*p++ = 0xc1;
		*(uint32*)p = siz/8;
		p += 4;

		// REP; MOVSQ
		*p++ = 0xf3;
		*p++ = 0x48;
		*p++ = 0xa5;
	}


	// call fn
	pcrel = fn - (p+5);
	if((int32)pcrel == pcrel) {
		// can use direct call with pc-relative offset
		// CALL fn
		*p++ = 0xe8;
		*(int32*)p = pcrel;
		p += 4;
	} else {
		// MOVQ $fn, CX  [64-bit immediate fn]
		*p++ = 0x48;
		*p++ = 0xb9;
		*(byte**)p = fn;
		p += 8;

		// CALL *CX
		*p++ = 0xff;
		*p++ = 0xd1;
	}

	// ADDQ $siz, SP
	*p++ = 0x48;
	*p++ = 0x81;
	*p++ = 0xc4;
	*(uint32*)p = siz;
	p += 4;

	// RET
	*p++ = 0xc3;

	if(p > q)
		throw("bad math in sys.closure");
}



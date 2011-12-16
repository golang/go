// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

#pragma textflag 7
// func closure(siz int32,
//	fn func(arg0, arg1, arg2 *ptr, callerpc uintptr, xxx) yyy,
//	arg0, arg1, arg2 *ptr) (func(xxx) yyy)
void
runtime·closure(int32 siz, byte *fn, byte *arg0)
{
	byte *p, *q, **ret;
	int32 i, n;
	int32 pcrel;

	if(siz < 0 || siz%4 != 0)
		runtime·throw("bad closure size");

	ret = (byte**)((byte*)&arg0 + siz);

	if(siz > 100) {
		// TODO(rsc): implement stack growth preamble?
		runtime·throw("closure too big");
	}

	// compute size of new fn.
	// must match code laid out below.
	n = 6+5+2+1;	// SUBL MOVL MOVL CLD
	if(siz <= 4*4)
		n += 1*siz/4;	// MOVSL MOVSL...
	else
		n += 6+2;	// MOVL REP MOVSL
	n += 5;	// CALL
	n += 6+1;	// ADDL RET

	// store args aligned after code, so gc can find them.
	n += siz;
	if(n%4)
		n += 4 - n%4;

	p = runtime·mal(n);
	*ret = p;
	q = p + n - siz;

	if(siz > 0) {
		runtime·memmove(q, (byte*)&arg0, siz);

		// SUBL $siz, SP
		*p++ = 0x81;
		*p++ = 0xec;
		*(uint32*)p = siz;
		p += 4;

		// MOVL $q, SI
		*p++ = 0xbe;
		*(byte**)p = q;
		p += 4;

		// MOVL SP, DI
		*p++ = 0x89;
		*p++ = 0xe7;

		// CLD
		*p++ = 0xfc;

		if(siz <= 4*4) {
			for(i=0; i<siz; i+=4) {
				// MOVSL
				*p++ = 0xa5;
			}
		} else {
			// MOVL $(siz/4), CX  [32-bit immediate siz/4]
			*p++ = 0xc7;
			*p++ = 0xc1;
			*(uint32*)p = siz/4;
			p += 4;

			// REP; MOVSL
			*p++ = 0xf3;
			*p++ = 0xa5;
		}
	}

	// call fn
	pcrel = fn - (p+5);
	// direct call with pc-relative offset
	// CALL fn
	*p++ = 0xe8;
	*(int32*)p = pcrel;
	p += 4;

	// ADDL $siz, SP
	*p++ = 0x81;
	*p++ = 0xc4;
	*(uint32*)p = siz;
	p += 4;

	// RET
	*p++ = 0xc3;

	if(p > q)
		runtime·throw("bad math in sys.closure");
}

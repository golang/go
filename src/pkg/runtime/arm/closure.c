// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

/*
	There are two bits of magic:
	- The signature of the compiler generated function uses two stack frames
	as arguments (callerpc separates these frames)
	- size determines how many arguments runtime.closure actually has
	starting at arg0.

	Example closure with 3 captured variables:
	func closure(siz int32,
	fn func(arg0, arg1, arg2 *ptr, callerpc uintptr, xxx) yyy,
		arg0, arg1, arg2 *ptr) (func(xxx) yyy)

	Code generated:
	src R0
	dst R1
	end R3
	tmp R4
	frame = siz+4

//skip loop for 0 size closures
		MOVW.W	R14,-frame(R13)

		MOVW	$vars(PC), R0
		MOVW	$4(SP), R1
		MOVW	$siz(R0), R3
loop:		MOVW.P	4(R0), R4
		MOVW.P	R4, 4(R1)
		CMP		R0, R3
		BNE		loop

		MOVW	8(PC), R0
		BL		(R0)			// 2 words
		MOVW.P	frame(R13),R15
fptr:		WORD	*fn
vars:		WORD	arg0
		WORD	arg1
		WORD	arg2
*/

extern void cacheflush(byte* start, byte* end);

#pragma textflag 7
void
·closure(int32 siz, byte *fn, byte *arg0)
{
	byte *p, *q, **ret;
	uint32 *pc;
	int32 n;

	if(siz < 0 || siz%4 != 0)
		throw("bad closure size");

	ret = (byte**)((byte*)&arg0 + siz);

	if(siz > 100) {
		// TODO(kaib): implement stack growth preamble?
		throw("closure too big");
	}

	// size of new fn.
	// must match code laid out below.
	if (siz > 0)
		n = 6 * 4 + 7 * 4;
	else
		n = 6 * 4;

	// store args aligned after code, so gc can find them.
	n += siz;

	p = mal(n);
	*ret = p;
	q = p + n - siz;

	pc = (uint32*)p;

	//	MOVW.W	R14,-frame(R13)
	*pc++ = 0xe52de000 | (siz + 4);

	if(siz > 0) {
		mcpy(q, (byte*)&arg0, siz);

		//	MOVW	$vars(PC), R0
		*pc = 0xe28f0000 | (int32)(q - (byte*)pc - 8);
		pc++;

		//	MOVW	$4(SP), R1
		*pc++ = 0xe28d1004;

		//	MOVW	$siz(R0), R3
		*pc++ = 0xe2803000 | siz;

		//	MOVW.P	4(R0), R4
		*pc++ = 0xe4904004;
		//	MOVW.P	R4, 4(R1)
		*pc++ = 0xe4814004;
		//	CMP		R0, R3
		*pc++ = 0xe1530000;
		//	BNE		loop
		*pc++ = 0x1afffffb;
	}

	//	MOVW	fptr(PC), R0
	*pc = 0xe59f0008 | (int32)((q - 4) -(byte*) pc - 8);
	pc++;

	//	BL		(R0)
	*pc++ = 0xe28fe000;
	*pc++ = 0xe280f000;

	//	MOVW.P	frame(R13),R15
	*pc++ = 0xe49df000 | (siz + 4);

	//	WORD	*fn
	*pc++ = (uint32)fn;

	p = (byte*)pc;

	if(p > q)
		throw("bad math in sys.closure");

	cacheflush(*ret, q+siz);
}


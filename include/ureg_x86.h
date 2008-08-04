// Inferno utils/libmach/ureg8.h
// http://code.google.com/p/inferno-os/source/browse/utils/libmach/ureg8.h
//
//	Copyright © 1994-1999 Lucent Technologies Inc.
//	Power PC support Copyright © 1995-2004 C H Forsyth (forsyth@terzarima.net).
//	Portions Copyright © 1997-1999 Vita Nuova Limited.
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com).
//	Revisions Copyright © 2000-2004 Lucent Technologies Inc. and others.
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

struct Ureg
{
	uint32	di;		/* general registers */
	uint32	si;		/* ... */
	uint32	bp;		/* ... */
	uint32	nsp;
	uint32	bx;		/* ... */
	uint32	dx;		/* ... */
	uint32	cx;		/* ... */
	uint32	ax;		/* ... */
	uint32	gs;		/* data segments */
	uint32	fs;		/* ... */
	uint32	es;		/* ... */
	uint32	ds;		/* ... */
	uint32	trap;		/* trap type */
	uint32	ecode;		/* error code (or zero) */
	uint32	pc;		/* pc */
	uint32	cs;		/* old context */
	uint32	flags;		/* old flags */
	union {
		uint32	usp;
		uint32	sp;
	};
	uint32	ss;		/* old stack segment */
};

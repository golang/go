// Inferno libmach/obj.c
// http://code.google.com/p/inferno-os/source/browse/utils/libmach/obj.c
//
// 	Copyright © 1994-1999 Lucent Technologies Inc.
// 	Power PC support Copyright © 1995-2004 C H Forsyth (forsyth@terzarima.net).
// 	Portions Copyright © 1997-1999 Vita Nuova Limited.
// 	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com).
// 	Revisions Copyright © 2000-2004 Lucent Technologies Inc. and others.
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

/*
 * obj.c
 * routines universal to all object files
 */
#include <u.h>
#include <libc.h>
#include <bio.h>
#include <ar.h>
#include <mach.h>
#include "obj.h"

int
isar(Biobuf *bp)
{
	int n;
	char magbuf[SARMAG];

	n = Bread(bp, magbuf, SARMAG);
	if(n == SARMAG && strncmp(magbuf, ARMAG, SARMAG) == 0)
		return 1;
	return 0;
}

/*
 * look for the next file in an archive
 */
int
nextar(Biobuf *bp, int offset, char *buf)
{
	struct ar_hdr a;
	int i, r;
	int32 arsize;

	if (offset&01)
		offset++;
	Bseek(bp, offset, 0);
	r = Bread(bp, &a, SAR_HDR);
	if(r != SAR_HDR)
		return 0;
	if(strncmp(a.fmag, ARFMAG, sizeof(a.fmag)))
		return -1;
	for(i=0; i<sizeof(a.name) && i<SARNAME && a.name[i] != ' '; i++)
		buf[i] = a.name[i];
	buf[i] = 0;
	arsize = strtol(a.size, 0, 0);
	if (arsize&1)
		arsize++;
	return arsize + SAR_HDR;
}

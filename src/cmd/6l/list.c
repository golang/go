// Inferno utils/6l/list.c
// http://code.google.com/p/inferno-os/source/browse/utils/6l/list.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
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

// Printing.

#include	"l.h"
#include	"../ld/lib.h"

void
listinit(void)
{
	listinit6();
	fmtinstall('I', Iconv);
}

int
Iconv(Fmt *fp)
{
	int i, n;
	uchar *p;
	char *s;
	Fmt fmt;
	
	n = fp->prec;
	fp->prec = 0;
	if(!(fp->flags&FmtPrec) || n < 0)
		return fmtstrcpy(fp, "%I");
	fp->flags &= ~FmtPrec;
	p = va_arg(fp->args, uchar*);

	// format into temporary buffer and
	// call fmtstrcpy to handle padding.
	fmtstrinit(&fmt);
	for(i=0; i<n; i++)
		fmtprint(&fmt, "%.2ux", *p++);
	s = fmtstrflush(&fmt);
	fmtstrcpy(fp, s);
	free(s);
	return 0;
}

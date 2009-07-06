// Derived from Inferno utils/5c/list.c
// http://code.google.com/p/inferno-os/source/browse/utils/5c/list.c
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

#include "gg.h"

// TODO(kaib): make 5g/list.c congruent with 5l/list.c

static	int	sconsize;
void
listinit(void)
{

	fmtinstall('A', Aconv);		// as
	fmtinstall('P', Pconv);		// Prog*
	fmtinstall('D', Dconv);		// Addr*
	fmtinstall('Y', Yconv);		// sconst
	fmtinstall('R', Rconv);		// register
	fmtinstall('M', Mconv);		// names
}

int
Pconv(Fmt *fp)
{
	char str[STRINGSZ];
	Prog *p;

	p = va_arg(fp->args, Prog*);
	sconsize = 8;
	switch(p->as) {
	default:
		snprint(str, sizeof(str), "%.4ld (%4ld) %-7A %D,%D",
			p->loc, p->lineno, p->as, &p->from, &p->to);
		break;

	case ADATA:
		sconsize = p->from.scale;
		snprint(str, sizeof(str), "%.4ld (%4ld) %-7A %D/%d,%D",
			p->loc, p->lineno, p->as, &p->from, sconsize, &p->to);
		break;

	case ATEXT:
		snprint(str, sizeof(str), "%.4ld (%4ld) %-7A %D,%lD",
			p->loc, p->lineno, p->as, &p->from, &p->to);
		break;
	}
	return fmtstrcpy(fp, str);
}

int
Dconv(Fmt *fp)
{
	char str[100], s[100];
	Addr *a;
	int i;
	uint32 d1, d2;

	a = va_arg(fp->args, Addr*);
	i = a->type;
	switch(i) {

	default:
		if(a->type == D_OREG)
			snprint(str, sizeof(str), "$%d(R%d)", a->offset, a->reg);
		else
			snprint(str, sizeof(str), "R%d", a->reg);
		break;

	case D_NONE:
		str[0] = 0;
		break;

	case D_BRANCH:
		snprint(str, sizeof(str), "%d", a->branch->loc);
		break;

	case D_CONST:
		if(a->reg != NREG)
			sprint(str, "$%M(R%d)", a, a->reg);
		else
			sprint(str, "$%M", a);
		break;

	case D_CONST2:
		sprint(str, "$%d-%d", a->offset, a->offset2);
		break;

	case D_FCONST:
		snprint(str, sizeof(str), "$(%.17e)", a->dval);
		break;

	case D_SCONST:
		snprint(str, sizeof(str), "$\"%Y\"", a->sval);
		break;

		// TODO(kaib): Add back
//	case D_ADDR:
//		a->type = a->index;
//		a->index = D_NONE;
//		snprint(str, sizeof(str), "$%D", a);
//		a->index = a->type;
//		a->type = D_ADDR;
//		goto conv;
	}
conv:
	return fmtstrcpy(fp, str);
}

int
Aconv(Fmt *fp)
{
	int i;

	i = va_arg(fp->args, int);
	return fmtstrcpy(fp, anames[i]);
}


int
Yconv(Fmt *fp)
{
	int i, c;
	char str[30], *p, *a;

	a = va_arg(fp->args, char*);
	p = str;
	for(i=0; i<sconsize; i++) {
		c = a[i] & 0xff;
		if((c >= 'a' && c <= 'z') ||
		   (c >= 'A' && c <= 'Z') ||
		   (c >= '0' && c <= '9')) {
			*p++ = c;
			continue;
		}
		*p++ = '\\';
		switch(c) {
		default:
			if(c < 040 || c >= 0177)
				break;	/* not portable */
			p[-1] = c;
			continue;
		case 0:
			*p++ = 'z';
			continue;
		case '\\':
		case '"':
			*p++ = c;
			continue;
		case '\n':
			*p++ = 'n';
			continue;
		case '\t':
			*p++ = 't';
			continue;
		}
		*p++ = (c>>6) + '0';
		*p++ = ((c>>3) & 7) + '0';
		*p++ = (c & 7) + '0';
	}
	*p = 0;
	return fmtstrcpy(fp, str);
}

int
Rconv(Fmt *fp)
{
	int r;
	char str[30];

	r = va_arg(fp->args, int);
	snprint(str, sizeof(str), "R%d", r);
	return fmtstrcpy(fp, str);
}

int
Mconv(Fmt *fp)
{
	char str[STRINGSZ];
	Addr *a;

	a = va_arg(fp->args, Addr*);
	switch(a->name) {
	case D_EXTERN:
		snprint(str, sizeof(str), "%S+%d(SB)", a->sym, a->offset);
		break;

	case D_STATIC:
		snprint(str, sizeof(str), "%S<>+%d(SB)", a->sym, a->offset);
		break;

	case D_AUTO:
		snprint(str, sizeof(str), "%S+%d(SP)", a->sym, a->offset);
		break;

	case D_PARAM:
		snprint(str, sizeof(str), "%S+%d(FP)", a->sym, a->offset);
		break;
	}
	return fmtstrcpy(fp, str);
}

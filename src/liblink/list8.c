// Inferno utils/8c/list.c
// http://code.google.com/p/inferno-os/source/browse/utils/8c/list.c
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

#include <u.h>
#include <libc.h>
#include <bio.h>
#include <link.h>
#include "../cmd/8l/8.out.h"

static int	Aconv(Fmt *fp);
static int	Dconv(Fmt *fp);
static int	Pconv(Fmt *fp);
static int	Rconv(Fmt *fp);
static int	DSconv(Fmt *fp);

enum
{
	STRINGSZ = 1000
};

#pragma	varargck	type	"$"	char*

void
listinit8(void)
{
	fmtinstall('A', Aconv);
	fmtinstall('D', Dconv);
	fmtinstall('P', Pconv);
	fmtinstall('R', Rconv);

	// for internal use
	fmtinstall('$', DSconv);
}

static	Prog*	bigP;

static int
Pconv(Fmt *fp)
{
	char str[STRINGSZ];
	Prog *p;

	p = va_arg(fp->args, Prog*);
	bigP = p;
	switch(p->as) {
	case ADATA:
		sprint(str, "%.5lld (%L)	%A	%D/%d,%D",
			p->pc, p->lineno, p->as, &p->from, p->from.scale, &p->to);
		break;

	case ATEXT:
		if(p->from.scale) {
			sprint(str, "%.5lld (%L)	%A	%D,%d,%lD",
				p->pc, p->lineno, p->as, &p->from, p->from.scale, &p->to);
			break;
		}
		sprint(str, "%.5lld (%L)	%A	%D,%lD",
			p->pc, p->lineno, p->as, &p->from, &p->to);
		break;

	default:
		sprint(str, "%.5lld (%L)	%A	%D,%D",
			p->pc, p->lineno, p->as, &p->from, &p->to);
		break;
	}
	bigP = nil;
	return fmtstrcpy(fp, str);
}

static int
Aconv(Fmt *fp)
{
	int i;

	i = va_arg(fp->args, int);
	return fmtstrcpy(fp, anames8[i]);
}

static int
Dconv(Fmt *fp)
{
	char str[STRINGSZ], s[STRINGSZ];
	Addr *a;
	int i;

	a = va_arg(fp->args, Addr*);
	i = a->type;

	if(fp->flags & FmtLong) {
		if(i == D_CONST2)
			sprint(str, "$%lld-%d", a->offset, a->offset2);
		else {
			// ATEXT dst is not constant
			sprint(str, "!!%D", a);
		}
		goto brk;
	}

	if(i >= D_INDIR) {
		if(a->offset)
			sprint(str, "%lld(%R)", a->offset, i-D_INDIR);
		else
			sprint(str, "(%R)", i-D_INDIR);
		goto brk;
	}
	switch(i) {
	default:
		if(a->offset)
			sprint(str, "$%lld,%R", a->offset, i);
		else
			sprint(str, "%R", i);
		break;

	case D_NONE:
		str[0] = 0;
		break;

	case D_BRANCH:
		if(a->sym != nil)
			sprint(str, "%s(SB)", a->sym->name);
		else if(bigP != nil && bigP->pcond != nil)
			sprint(str, "%lld", bigP->pcond->pc);
		else if(a->u.branch != nil)
			sprint(str, "%lld", a->u.branch->pc);
		else
			sprint(str, "%lld(PC)", a->offset);
		break;

	case D_EXTERN:
		sprint(str, "%s+%lld(SB)", a->sym->name, a->offset);
		break;

	case D_STATIC:
		sprint(str, "%s<>+%lld(SB)", a->sym->name, a->offset);
		break;

	case D_AUTO:
		if(a->sym)
			sprint(str, "%s+%lld(SP)", a->sym->name, a->offset);
		else
			sprint(str, "%lld(SP)", a->offset);
		break;

	case D_PARAM:
		if(a->sym)
			sprint(str, "%s+%lld(FP)", a->sym->name, a->offset);
		else
			sprint(str, "%lld(FP)", a->offset);
		break;

	case D_CONST:
		sprint(str, "$%lld", a->offset);
		break;

	case D_CONST2:
		if(!(fp->flags & FmtLong)) {
			// D_CONST2 outside of ATEXT should not happen
			sprint(str, "!!$%lld-%d", a->offset, a->offset2);
		}
		break;

	case D_FCONST:
		sprint(str, "$(%.17g)", a->u.dval);
		break;

	case D_SCONST:
		sprint(str, "$\"%$\"", a->u.sval);
		break;

	case D_ADDR:
		a->type = a->index;
		a->index = D_NONE;
		sprint(str, "$%D", a);
		a->index = a->type;
		a->type = D_ADDR;
		goto conv;
	}
brk:
	if(a->index != D_NONE) {
		sprint(s, "(%R*%d)", (int)a->index, (int)a->scale);
		strcat(str, s);
	}
conv:
	return fmtstrcpy(fp, str);
}

static char*	regstr[] =
{
	"AL",	/* [D_AL] */
	"CL",
	"DL",
	"BL",
	"AH",
	"CH",
	"DH",
	"BH",

	"AX",	/* [D_AX] */
	"CX",
	"DX",
	"BX",
	"SP",
	"BP",
	"SI",
	"DI",

	"F0",	/* [D_F0] */
	"F1",
	"F2",
	"F3",
	"F4",
	"F5",
	"F6",
	"F7",

	"CS",	/* [D_CS] */
	"SS",
	"DS",
	"ES",
	"FS",
	"GS",

	"GDTR",	/* [D_GDTR] */
	"IDTR",	/* [D_IDTR] */
	"LDTR",	/* [D_LDTR] */
	"MSW",	/* [D_MSW] */
	"TASK",	/* [D_TASK] */

	"CR0",	/* [D_CR] */
	"CR1",
	"CR2",
	"CR3",
	"CR4",
	"CR5",
	"CR6",
	"CR7",

	"DR0",	/* [D_DR] */
	"DR1",
	"DR2",
	"DR3",
	"DR4",
	"DR5",
	"DR6",
	"DR7",

	"TR0",	/* [D_TR] */
	"TR1",
	"TR2",
	"TR3",
	"TR4",
	"TR5",
	"TR6",
	"TR7",

	"X0",	/* [D_X0] */
	"X1",
	"X2",
	"X3",
	"X4",
	"X5",
	"X6",
	"X7",

	"TLS",	/* [D_TLS] */
	"NONE",	/* [D_NONE] */
};

static int
Rconv(Fmt *fp)
{
	char str[STRINGSZ];
	int r;

	r = va_arg(fp->args, int);
	if(r >= D_AL && r <= D_NONE)
		sprint(str, "%s", regstr[r-D_AL]);
	else
		sprint(str, "gok(%d)", r);

	return fmtstrcpy(fp, str);
}

static int
DSconv(Fmt *fp)
{
	int i, c;
	char str[STRINGSZ], *p, *a;

	a = va_arg(fp->args, char*);
	p = str;
	for(i=0; i<sizeof(double); i++) {
		c = a[i] & 0xff;
		if(c >= 'a' && c <= 'z' ||
		   c >= 'A' && c <= 'Z' ||
		   c >= '0' && c <= '9') {
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

// Inferno utils/6c/list.c
// http://code.google.com/p/inferno-os/source/browse/utils/6c/list.c
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

#define EXTERN
#include "gc.h"

void
listinit(void)
{

	fmtinstall('A', Aconv);
	fmtinstall('B', Bconv);
	fmtinstall('P', Pconv);
	fmtinstall('S', Sconv);
	fmtinstall('D', Dconv);
	fmtinstall('R', Rconv);
}

int
Bconv(Fmt *fp)
{
	char str[STRINGSZ], ss[STRINGSZ], *s;
	Bits bits;
	int i;

	str[0] = 0;
	bits = va_arg(fp->args, Bits);
	while(bany(&bits)) {
		i = bnum(bits);
		if(str[0])
			strcat(str, " ");
		if(var[i].sym == S) {
			sprint(ss, "$%lld", var[i].offset);
			s = ss;
		} else
			s = var[i].sym->name;
		if(strlen(str) + strlen(s) + 1 >= STRINGSZ)
			break;
		strcat(str, s);
		bits.b[i/32] &= ~(1L << (i%32));
	}
	return fmtstrcpy(fp, str);
}

int
Pconv(Fmt *fp)
{
	char str[STRINGSZ];
	Prog *p;

	p = va_arg(fp->args, Prog*);
	switch(p->as) {
	case ADATA:
		sprint(str, "(%L)	%A	%D/%d,%D",
			p->lineno, p->as, &p->from, p->from.scale, &p->to);
		break;

	case ATEXT:
		if(p->from.scale) {
			sprint(str, "(%L)	%A	%D,%d,%lD",
				p->lineno, p->as, &p->from, p->from.scale, &p->to);
			break;
		}
		sprint(str, "(%L)	%A	%D,%lD",
			p->lineno, p->as, &p->from, &p->to);
		break;

	default:
		sprint(str, "(%L)	%A	%D,%D",
			p->lineno, p->as, &p->from, &p->to);
		break;
	}
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
Dconv(Fmt *fp)
{
	char str[STRINGSZ], s[STRINGSZ];
	Adr *a;
	int i;

	a = va_arg(fp->args, Adr*);
	i = a->type;

	if(fp->flags & FmtLong) {
		if(i == D_CONST)
			sprint(str, "$%lld-%lld", a->offset&0xffffffffLL, a->offset>>32);
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
		sprint(str, "%lld", a->offset);
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

	case D_FCONST:
		sprint(str, "$(%.17e)", a->dval);
		break;

	case D_SCONST:
		sprint(str, "$\"%S\"", a->sval);
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

char*	regstr[] =
{
	"AL",	/* [D_AL] */
	"CL",
	"DL",
	"BL",
	"SPB",
	"BPB",
	"SIB",
	"DIB",
	"R8B",
	"R9B",
	"R10B",
	"R11B",
	"R12B",
	"R13B",
	"R14B",
	"R15B",

	"AX",	/* [D_AX] */
	"CX",
	"DX",
	"BX",
	"SP",
	"BP",
	"SI",
	"DI",
	"R8",
	"R9",
	"R10",
	"R11",
	"R12",
	"R13",
	"R14",
	"R15",

	"AH",
	"CH",
	"DH",
	"BH",

	"F0",	/* [D_F0] */
	"F1",
	"F2",
	"F3",
	"F4",
	"F5",
	"F6",
	"F7",

	"M0",
	"M1",
	"M2",
	"M3",
	"M4",
	"M5",
	"M6",
	"M7",

	"X0",
	"X1",
	"X2",
	"X3",
	"X4",
	"X5",
	"X6",
	"X7",
	"X8",
	"X9",
	"X10",
	"X11",
	"X12",
	"X13",
	"X14",
	"X15",

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
	"CR8",
	"CR9",
	"CR10",
	"CR11",
	"CR12",
	"CR13",
	"CR14",
	"CR15",

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

	"NONE",	/* [D_NONE] */
};

int
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

int
Sconv(Fmt *fp)
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

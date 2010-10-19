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

static	Prog*	bigP;

void
listinit(void)
{

	fmtinstall('R', Rconv);
	fmtinstall('A', Aconv);
	fmtinstall('D', Dconv);
	fmtinstall('S', Sconv);
	fmtinstall('P', Pconv);
	fmtinstall('I', Iconv);
}

int
Pconv(Fmt *fp)
{
	Prog *p;

	p = va_arg(fp->args, Prog*);
	bigP = p;
	switch(p->as) {
	case ATEXT:
		if(p->from.scale) {
			fmtprint(fp, "(%d)	%A	%D,%d,%D",
				p->line, p->as, &p->from, p->from.scale, &p->to);
			break;
		}
	default:
		fmtprint(fp, "(%d)	%A	%D,%D",
			p->line, p->as, &p->from, &p->to);
		break;
	case ADATA:
	case AINIT_:
	case ADYNT_:
		fmtprint(fp, "(%d)	%A	%D/%d,%D",
			p->line, p->as, &p->from, p->from.scale, &p->to);
		break;
	}
	bigP = P;
	return 0;
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
		if(i != D_CONST) {
			// ATEXT dst is not constant
			snprint(str, sizeof(str), "!!%D", a);
			goto brk;
		}
		parsetextconst(a->offset);
		if(textarg == 0) {
			snprint(str, sizeof(str), "$%lld", textstksiz);
			goto brk;
		}
		snprint(str, sizeof(str), "$%lld-%lld", textstksiz, textarg);
		goto brk;
	}

	if(i >= D_INDIR) {
		if(a->offset)
			snprint(str, sizeof(str), "%lld(%R)", a->offset, i-D_INDIR);
		else
			snprint(str, sizeof(str), "(%R)", i-D_INDIR);
		goto brk;
	}
	switch(i) {

	default:
		if(a->offset)
			snprint(str, sizeof(str), "$%lld,%R", a->offset, i);
		else
			snprint(str, sizeof(str), "%R", i);
		break;

	case D_NONE:
		str[0] = 0;
		break;

	case D_BRANCH:
		if(bigP != P && bigP->pcond != P)
			if(a->sym != S)
				snprint(str, sizeof(str), "%llux+%s", bigP->pcond->pc,
					a->sym->name);
			else
				snprint(str, sizeof(str), "%llux", bigP->pcond->pc);
		else
			snprint(str, sizeof(str), "%lld(PC)", a->offset);
		break;

	case D_EXTERN:
		if(a->sym) {
			snprint(str, sizeof(str), "%s+%lld(SB)", a->sym->name, a->offset);
			break;
		}
		snprint(str, sizeof(str), "!!noname!!+%lld(SB)", a->offset);
		break;

	case D_STATIC:
		if(a->sym) {
			snprint(str, sizeof(str), "%s<%d>+%lld(SB)", a->sym->name,
				a->sym->version, a->offset);
			break;
		}
		snprint(str, sizeof(str), "!!noname!!<999>+%lld(SB)", a->offset);
		break;

	case D_AUTO:
		if(a->sym) {
			snprint(str, sizeof(str), "%s+%lld(SP)", a->sym->name, a->offset);
			break;
		}
		snprint(str, sizeof(str), "!!noname!!+%lld(SP)", a->offset);
		break;

	case D_PARAM:
		if(a->sym) {
			snprint(str, sizeof(str), "%s+%lld(%s)", a->sym->name, a->offset, paramspace);
			break;
		}
		snprint(str, sizeof(str), "!!noname!!+%lld(%s)", a->offset, paramspace);
		break;

	case D_CONST:
		snprint(str, sizeof(str), "$%lld", a->offset);
		break;

	case D_FCONST:
		snprint(str, sizeof(str), "$(%.8ux,%.8ux)", a->ieee.h, a->ieee.l);
		break;

	case D_SCONST:
		snprint(str, sizeof(str), "$\"%S\"", a->scon);
		break;

	case D_ADDR:
		a->type = a->index;
		a->index = D_NONE;
		snprint(str, sizeof(str), "$%D", a);
		a->index = a->type;
		a->type = D_ADDR;
		goto conv;
	}
brk:
	if(a->index != D_NONE) {
		snprint(s, sizeof(s), "(%R*%d)", a->index, a->scale);
		strcat(str, s);
	}
conv:
	fmtstrcpy(fp, str);
//	if(a->gotype)
//		fmtprint(fp, "«%s»", a->gotype->name);
	return 0;

}

char*	regstr[] =
{
	"AL",		/* [D_AL] */
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

	"AX",		/* [D_AX] */
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

	"F0",		/* [D_F0] */
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

	"CS",		/* [D_CS] */
	"SS",
	"DS",
	"ES",
	"FS",
	"GS",

	"GDTR",		/* [D_GDTR] */
	"IDTR",		/* [D_IDTR] */
	"LDTR",		/* [D_LDTR] */
	"MSW",		/* [D_MSW] */
	"TASK",		/* [D_TASK] */

	"CR0",		/* [D_CR] */
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

	"DR0",		/* [D_DR] */
	"DR1",
	"DR2",
	"DR3",
	"DR4",
	"DR5",
	"DR6",
	"DR7",

	"TR0",		/* [D_TR] */
	"TR1",
	"TR2",
	"TR3",
	"TR4",
	"TR5",
	"TR6",
	"TR7",

	"NONE",		/* [D_NONE] */
};

int
Rconv(Fmt *fp)
{
	char str[STRINGSZ];
	int r;

	r = va_arg(fp->args, int);
	if(r >= D_AL && r <= D_NONE)
		snprint(str, sizeof(str), "%s", regstr[r-D_AL]);
	else
		snprint(str, sizeof(str), "gok(%d)", r);

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

void
diag(char *fmt, ...)
{
	char buf[STRINGSZ], *tn, *sep;
	va_list arg;

	tn = "";
	sep = "";
	if(cursym != S) {
		tn = cursym->name;
		sep = ": ";
	}
	va_start(arg, fmt);
	vseprint(buf, buf+sizeof(buf), fmt, arg);
	va_end(arg);
	print("%s%s%s\n", tn, sep, buf);

	nerrors++;
	if(nerrors > 20) {
		print("too many errors\n");
		errorexit();
	}
}

void
parsetextconst(vlong arg)
{
	textstksiz = arg & 0xffffffffLL;
	if(textstksiz & 0x80000000LL)
		textstksiz = -(-textstksiz & 0xffffffffLL);

	textarg = (arg >> 32) & 0xffffffffLL;
	if(textarg & 0x80000000LL)
		textarg = 0;
	textarg = (textarg+7) & ~7LL;
}

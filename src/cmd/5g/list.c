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

#include <u.h>
#include <libc.h>
#include "gg.h"

// TODO(kaib): make 5g/list.c congruent with 5l/list.c

static	int	sconsize;
void
listinit(void)
{

	fmtinstall('A', Aconv);		// as
	fmtinstall('C', Cconv);		// conditional execution bit
	fmtinstall('P', Pconv);			// Prog*
	fmtinstall('D', Dconv);		// Addr*
	fmtinstall('Y', Yconv);		// sconst
	fmtinstall('R', Rconv);		// register
	fmtinstall('M', Mconv);		// names
}

int
Pconv(Fmt *fp)
{
	char str[STRINGSZ], str1[STRINGSZ];
	Prog *p;

	p = va_arg(fp->args, Prog*);
	sconsize = 8;
	switch(p->as) {
	default:
		snprint(str1, sizeof(str1), "%A%C", p->as, p->scond);
		if(p->reg == NREG && p->as != AGLOBL)
			snprint(str, sizeof(str), "%.4d (%L) %-7s	%D,%D", 
				p->loc, p->lineno, str1, &p->from, &p->to);
		else
		if (p->from.type != D_FREG) {
			snprint(str, sizeof(str), "%.4d (%L) %-7s	%D,R%d,%D", 
				p->loc, p->lineno, str1, &p->from, p->reg, &p->to);
		} else
			snprint(str, sizeof(str), "%.4d (%L) %-7A%C	%D,F%d,%D",
				p->loc, p->lineno, p->as, p->scond, &p->from, p->reg, &p->to);
		break;

	case ADATA:
		snprint(str, sizeof(str), "%.4d (%L) %-7A	%D/%d,%D",
			p->loc, p->lineno, p->as, &p->from, p->reg, &p->to);
		break;
	}
	return fmtstrcpy(fp, str);
}

int
Dconv(Fmt *fp)
{
	char str[STRINGSZ];
	const char *op;
	Addr *a;
	int i;
	int32 v;

	a = va_arg(fp->args, Addr*);
	if(a == A) {
		sprint(str, "<nil>");
		goto conv;
	}
	i = a->type;
	switch(i) {

	default:
		sprint(str, "GOK-type(%d)", a->type);
		break;

	case D_NONE:
		str[0] = 0;
		if(a->name != D_NONE || a->reg != NREG || a->sym != S)
			sprint(str, "%M(R%d)(NONE)", a, a->reg);
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

	case D_SHIFT:
		v = a->offset;
		op = &"<<>>->@>"[(((v>>5) & 3) << 1)];
		if(v & (1<<4))
			sprint(str, "R%d%c%cR%d", v&15, op[0], op[1], (v>>8)&15);
		else
			sprint(str, "R%d%c%c%d", v&15, op[0], op[1], (v>>7)&31);
		if(a->reg != NREG)
			sprint(str+strlen(str), "(R%d)", a->reg);
		break;

	case D_OCONST:
		sprint(str, "$*$%M", a);
		if(a->reg != NREG)
			sprint(str, "%M(R%d)(CONST)", a, a->reg);
		break;

	case D_OREG:
		if(a->reg != NREG)
			sprint(str, "%M(R%d)", a, a->reg);
		else
			sprint(str, "%M", a);
		break;

	case D_REG:
		sprint(str, "R%d", a->reg);
		if(a->name != D_NONE || a->sym != S)
			sprint(str, "%M(R%d)(REG)", a, a->reg);
		break;

	case D_REGREG:
		sprint(str, "(R%d,R%d)", a->reg, (int)a->offset);
		if(a->name != D_NONE || a->sym != S)
			sprint(str, "%M(R%d)(REG)", a, a->reg);
		break;

	case D_REGREG2:
		sprint(str, "R%d,R%d", a->reg, (int)a->offset);
		if(a->name != D_NONE || a->sym != S)
			sprint(str, "%M(R%d)(REG)", a, a->reg);
		break;

	case D_FREG:
		sprint(str, "F%d", a->reg);
		if(a->name != D_NONE || a->sym != S)
			sprint(str, "%M(R%d)(REG)", a, a->reg);
		break;

	case D_BRANCH:
		if(a->u.branch == P || a->u.branch->loc == 0) {
			if(a->sym != S)
				sprint(str, "%s+%d(APC)", a->sym->name, a->offset);
			else
				sprint(str, "%d(APC)", a->offset);
		} else
			if(a->sym != S)
				sprint(str, "%s+%d(APC)", a->sym->name, a->u.branch->loc);
			else
				sprint(str, "%d(APC)", a->u.branch->loc);
		break;

	case D_FCONST:
		snprint(str, sizeof(str), "$(%.17e)", a->u.dval);
		break;

	case D_SCONST:
		snprint(str, sizeof(str), "$\"%Y\"", a->u.sval);
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
	fmtstrcpy(fp, str);
	if(a->gotype)
		fmtprint(fp, "{%s}", a->gotype->name);
	return 0;
}

int
Aconv(Fmt *fp)
{
	int i;

	i = va_arg(fp->args, int);
	return fmtstrcpy(fp, anames[i]);
}

char*	strcond[16] =
{
	".EQ",
	".NE",
	".HS",
	".LO",
	".MI",
	".PL",
	".VS",
	".VC",
	".HI",
	".LS",
	".GE",
	".LT",
	".GT",
	".LE",
	"",
	".NV"
};

int
Cconv(Fmt *fp)
{
	char s[STRINGSZ];
	int c;

	c = va_arg(fp->args, int);
	strcpy(s, strcond[c & C_SCOND]);
	if(c & C_SBIT)
		strcat(s, ".S");
	if(c & C_PBIT)
		strcat(s, ".P");
	if(c & C_WBIT)
		strcat(s, ".W");
	if(c & C_UBIT)		/* ambiguous with FBIT */
		strcat(s, ".U");
	return fmtstrcpy(fp, s);
}

int
Yconv(Fmt *fp)
{
	int i, c;
	char str[STRINGSZ], *p, *a;

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
	char str[STRINGSZ];

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
	default:
		snprint(str, sizeof(str),  "GOK-name(%d)", a->name);
		break;

	case D_NONE:
		snprint(str, sizeof(str), "%d", a->offset);
		break;

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

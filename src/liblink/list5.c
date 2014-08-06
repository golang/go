// Inferno utils/5c/list.c
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
#include <bio.h>
#include <link.h>
#include "../cmd/5l/5.out.h"

enum
{
	STRINGSZ = 1000
};

static int	Aconv(Fmt *fp);
static int	Dconv(Fmt *fp);
static int	Mconv(Fmt *fp);
static int	Pconv(Fmt *fp);
static int	Rconv(Fmt *fp);
static int	RAconv(Fmt *fp);
static int	DSconv(Fmt *fp);
static int	DRconv(Fmt*);

#pragma	varargck	type	"$"	char*
#pragma	varargck	type	"M"	Addr*
#pragma	varargck	type	"@"	Addr*

void
listinit5(void)
{
	fmtinstall('A', Aconv);
	fmtinstall('D', Dconv);
	fmtinstall('P', Pconv);
	fmtinstall('R', Rconv);

	// for liblink internal use
	fmtinstall('^', DRconv);

	// for internal use
	fmtinstall('$', DSconv);
	fmtinstall('M', Mconv);
	fmtinstall('@', RAconv);
}

static char *extra [] = {
	".EQ", ".NE", ".CS", ".CC",
	".MI", ".PL", ".VS", ".VC",
	".HI", ".LS", ".GE", ".LT",
	".GT", ".LE", "", ".NV",
};

static	Prog*	bigP;

static int
Pconv(Fmt *fp)
{
	char str[STRINGSZ], sc[20];
	Prog *p;
	int a, s;

	p = va_arg(fp->args, Prog*);
	bigP = p;
	a = p->as;
	s = p->scond;
	strcpy(sc, extra[s & C_SCOND]);
	if(s & C_SBIT)
		strcat(sc, ".S");
	if(s & C_PBIT)
		strcat(sc, ".P");
	if(s & C_WBIT)
		strcat(sc, ".W");
	if(s & C_UBIT)		/* ambiguous with FBIT */
		strcat(sc, ".U");
	if(a == AMOVM) {
		if(p->from.type == D_CONST)
			sprint(str, "%.5lld (%L)	%A%s	%@,%D", p->pc, p->lineno, a, sc, &p->from, &p->to);
		else
		if(p->to.type == D_CONST)
			sprint(str, "%.5lld (%L)	%A%s	%D,%@", p->pc, p->lineno, a, sc, &p->from, &p->to);
		else
			sprint(str, "%.5lld (%L)	%A%s	%D,%D", p->pc, p->lineno, a, sc, &p->from, &p->to);
	} else
	if(a == ADATA)
		sprint(str, "%.5lld (%L)	%A	%D/%d,%D", p->pc, p->lineno, a, &p->from, p->reg, &p->to);
	else
	if(p->as == ATEXT)
		sprint(str, "%.5lld (%L)	%A	%D,%d,%D", p->pc, p->lineno, a, &p->from, p->reg, &p->to);
	else
	if(p->reg == NREG)
		sprint(str, "%.5lld (%L)	%A%s	%D,%D", p->pc, p->lineno, a, sc, &p->from, &p->to);
	else
	if(p->from.type != D_FREG)
		sprint(str, "%.5lld (%L)	%A%s	%D,R%d,%D", p->pc, p->lineno, a, sc, &p->from, p->reg, &p->to);
	else
		sprint(str, "%.5lld (%L)	%A%s	%D,F%d,%D", p->pc, p->lineno, a, sc, &p->from, p->reg, &p->to);
	bigP = nil;
	return fmtstrcpy(fp, str);
}

static int
Aconv(Fmt *fp)
{
	char *s;
	int a;

	a = va_arg(fp->args, int);
	s = "???";
	if(a >= AXXX && a < ALAST)
		s = anames5[a];
	return fmtstrcpy(fp, s);
}

static int
Dconv(Fmt *fp)
{
	char str[STRINGSZ];
	Addr *a;
	const char *op;
	int v;

	a = va_arg(fp->args, Addr*);
	switch(a->type) {

	default:
		sprint(str, "GOK-type(%d)", a->type);
		break;

	case D_NONE:
		str[0] = 0;
		if(a->name != D_NONE || a->reg != NREG || a->sym != nil)
			sprint(str, "%M(R%d)(NONE)", a, a->reg);
		break;

	case D_CONST:
		if(a->reg != NREG)
			sprint(str, "$%M(R%d)", a, a->reg);
		else
			sprint(str, "$%M", a);
		break;

	case D_CONST2:
		sprint(str, "$%lld-%d", a->offset, a->offset2);
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

	case D_OREG:
		if(a->reg != NREG)
			sprint(str, "%M(R%d)", a, a->reg);
		else
			sprint(str, "%M", a);
		break;

	case D_REG:
		sprint(str, "R%d", a->reg);
		if(a->name != D_NONE || a->sym != nil)
			sprint(str, "%M(R%d)(REG)", a, a->reg);
		break;

	case D_FREG:
		sprint(str, "F%d", a->reg);
		if(a->name != D_NONE || a->sym != nil)
			sprint(str, "%M(R%d)(REG)", a, a->reg);
		break;

	case D_PSR:
		sprint(str, "PSR");
		if(a->name != D_NONE || a->sym != nil)
			sprint(str, "%M(PSR)(REG)", a);
		break;

	case D_BRANCH:
		if(a->sym != nil)
			sprint(str, "%s(SB)", a->sym->name);
		else if(bigP != nil && bigP->pcond != nil)
			sprint(str, "%lld", bigP->pcond->pc);
		else if(a->u.branch != nil)
			sprint(str, "%lld", a->u.branch->pc);
		else
			sprint(str, "%lld(PC)", a->offset/*-pc*/);
		break;

	case D_FCONST:
		sprint(str, "$%.17g", a->u.dval);
		break;

	case D_SCONST:
		sprint(str, "$\"%$\"", a->u.sval);
		break;
	}
	return fmtstrcpy(fp, str);
}

static int
RAconv(Fmt *fp)
{
	char str[STRINGSZ];
	Addr *a;
	int i, v;

	a = va_arg(fp->args, Addr*);
	sprint(str, "GOK-reglist");
	switch(a->type) {
	case D_CONST:
	case D_CONST2:
		if(a->reg != NREG)
			break;
		if(a->sym != nil)
			break;
		v = a->offset;
		strcpy(str, "");
		for(i=0; i<NREG; i++) {
			if(v & (1<<i)) {
				if(str[0] == 0)
					strcat(str, "[R");
				else
					strcat(str, ",R");
				sprint(strchr(str, 0), "%d", i);
			}
		}
		strcat(str, "]");
	}
	return fmtstrcpy(fp, str);
}

static int
DSconv(Fmt *fp)
{
	int i, c;
	char str[STRINGSZ], *p, *a;

	a = va_arg(fp->args, char*);
	p = str;
	for(i=0; i<NSNAME; i++) {
		c = a[i] & 0xff;
		if(c >= 'a' && c <= 'z' ||
		   c >= 'A' && c <= 'Z' ||
		   c >= '0' && c <= '9' ||
		   c == ' ' || c == '%') {
			*p++ = c;
			continue;
		}
		*p++ = '\\';
		switch(c) {
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
		case '\r':
			*p++ = 'r';
			continue;
		case '\f':
			*p++ = 'f';
			continue;
		}
		*p++ = (c>>6) + '0';
		*p++ = ((c>>3) & 7) + '0';
		*p++ = (c & 7) + '0';
	}
	*p = 0;
	return fmtstrcpy(fp, str);
}

static int
Rconv(Fmt *fp)
{
	int r;
	char str[STRINGSZ];

	r = va_arg(fp->args, int);
	sprint(str, "R%d", r);
	return fmtstrcpy(fp, str);
}

static int
DRconv(Fmt *fp)
{
	char *s;
	int a;

	a = va_arg(fp->args, int);
	s = "C_??";
	if(a >= C_NONE && a <= C_NCLASS)
		s = cnames5[a];
	return fmtstrcpy(fp, s);
}

static int
Mconv(Fmt *fp)
{
	char str[STRINGSZ];
	Addr *a;
	LSym *s;

	a = va_arg(fp->args, Addr*);
	s = a->sym;
	if(s == nil) {
		sprint(str, "%d", (int)a->offset);
		goto out;
	}
	switch(a->name) {
	default:
		sprint(str, "GOK-name(%d)", a->name);
		break;

	case D_NONE:
		sprint(str, "%lld", a->offset);
		break;

	case D_EXTERN:
		sprint(str, "%s+%d(SB)", s->name, (int)a->offset);
		break;

	case D_STATIC:
		sprint(str, "%s<>+%d(SB)", s->name, (int)a->offset);
		break;

	case D_AUTO:
		sprint(str, "%s-%d(SP)", s->name, (int)-a->offset);
		break;

	case D_PARAM:
		sprint(str, "%s+%d(FP)", s->name, (int)a->offset);
		break;
	}
out:
	return fmtstrcpy(fp, str);
}

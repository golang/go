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
#include "../runtime/funcdata.h"

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
	strcpy(sc, extra[(s & C_SCOND) ^ C_SCOND_XOR]);
	if(s & C_SBIT)
		strcat(sc, ".S");
	if(s & C_PBIT)
		strcat(sc, ".P");
	if(s & C_WBIT)
		strcat(sc, ".W");
	if(s & C_UBIT)		/* ambiguous with FBIT */
		strcat(sc, ".U");
	if(a == AMOVM) {
		if(p->from.type == TYPE_CONST)
			sprint(str, "%.5lld (%L)	%A%s	%@,%D", p->pc, p->lineno, a, sc, &p->from, &p->to);
		else
		if(p->to.type == TYPE_CONST)
			sprint(str, "%.5lld (%L)	%A%s	%D,%@", p->pc, p->lineno, a, sc, &p->from, &p->to);
		else
			sprint(str, "%.5lld (%L)	%A%s	%D,%D", p->pc, p->lineno, a, sc, &p->from, &p->to);
	} else
	if(a == ADATA)
		sprint(str, "%.5lld (%L)	%A	%D/%lld,%D", p->pc, p->lineno, a, &p->from, p->from3.offset, &p->to);
	else
	if(p->as == ATEXT)
		sprint(str, "%.5lld (%L)	%A	%D,%lld,%D", p->pc, p->lineno, a, &p->from, p->from3.offset, &p->to);
	else
	if(p->reg == 0)
		sprint(str, "%.5lld (%L)	%A%s	%D,%D", p->pc, p->lineno, a, sc, &p->from, &p->to);
	else
		sprint(str, "%.5lld (%L)	%A%s	%D,%R,%D", p->pc, p->lineno, a, sc, &p->from, p->reg, &p->to);
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

	case TYPE_NONE:
		str[0] = 0;
		if(a->name != TYPE_NONE || a->reg != 0 || a->sym != nil)
			sprint(str, "%M(%R)(NONE)", a, a->reg);
		break;

	case TYPE_CONST:
	case TYPE_ADDR:
		if(a->reg != 0)
			sprint(str, "$%M(%R)", a, a->reg);
		else
			sprint(str, "$%M", a);
		break;

	case TYPE_TEXTSIZE:
		if(a->u.argsize == ArgsSizeUnknown)
			sprint(str, "$%lld", a->offset);
		else
			sprint(str, "$%lld-%lld", a->offset, a->u.argsize);
		break;

	case TYPE_SHIFT:
		v = a->offset;
		op = &"<<>>->@>"[(((v>>5) & 3) << 1)];
		if(v & (1<<4))
			sprint(str, "R%d%c%cR%d", v&15, op[0], op[1], (v>>8)&15);
		else
			sprint(str, "R%d%c%c%d", v&15, op[0], op[1], (v>>7)&31);
		if(a->reg != 0)
			sprint(str+strlen(str), "(%R)", a->reg);
		break;

	case TYPE_MEM:
		if(a->reg != 0)
			sprint(str, "%M(%R)", a, a->reg);
		else
			sprint(str, "%M", a);
		break;

	case TYPE_REG:
		sprint(str, "%R", a->reg);
		if(a->name != TYPE_NONE || a->sym != nil)
			sprint(str, "%M(%R)(REG)", a, a->reg);
		break;

	case TYPE_BRANCH:
		if(a->sym != nil)
			sprint(str, "%s(SB)", a->sym->name);
		else if(bigP != nil && bigP->pcond != nil)
			sprint(str, "%lld", bigP->pcond->pc);
		else if(a->u.branch != nil)
			sprint(str, "%lld", a->u.branch->pc);
		else
			sprint(str, "%lld(PC)", a->offset/*-pc*/);
		break;

	case TYPE_FCONST:
		sprint(str, "$%.17g", a->u.dval);
		break;

	case TYPE_SCONST:
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
	case TYPE_CONST:
		if(a->reg != 0)
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

	r = va_arg(fp->args, int);
	if(r == 0)
		return fmtstrcpy(fp, "NONE");
	if(REG_R0 <= r && r <= REG_R15)
		return fmtprint(fp, "R%d", r-REG_R0);
	if(REG_F0 <= r && r <= REG_F15)
		return fmtprint(fp, "F%d", r-REG_F0);

	switch(r) {
	case REG_FPSR:
		return fmtstrcpy(fp, "FPSR");
	case REG_FPCR:
		return fmtstrcpy(fp, "FPCR");
	case REG_CPSR:
		return fmtstrcpy(fp, "CPSR");
	case REG_SPSR:
		return fmtstrcpy(fp, "SPSR");
	}

	return fmtprint(fp, "badreg(%d)", r);
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

	case NAME_NONE:
		sprint(str, "%lld", a->offset);
		break;

	case NAME_EXTERN:
		sprint(str, "%s+%d(SB)", s->name, (int)a->offset);
		break;

	case NAME_STATIC:
		sprint(str, "%s<>+%d(SB)", s->name, (int)a->offset);
		break;

	case NAME_AUTO:
		sprint(str, "%s-%d(SP)", s->name, (int)-a->offset);
		break;

	case NAME_PARAM:
		sprint(str, "%s+%d(FP)", s->name, (int)a->offset);
		break;
	}
out:
	return fmtstrcpy(fp, str);
}

// cmd/9l/list.c from Vita Nuova.
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2008 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2008 Lucent Technologies Inc. and others
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
#include "../cmd/9l/9.out.h"
#include "../runtime/funcdata.h"

enum
{
	STRINGSZ	= 1000,
};

static int	Aconv(Fmt*);
static int	Dconv(Fmt*);
static int	Pconv(Fmt*);
static int	Rconv(Fmt*);
static int	DSconv(Fmt*);
static int	Mconv(Fmt*);
static int	DRconv(Fmt*);

//
// Format conversions
//	%A int		Opcodes (instruction mnemonics)
//
//	%D Addr*	Addresses (instruction operands)
//
//	%P Prog*	Instructions
//
//	%R int		Registers
//
//	%$ char*	String constant addresses (for internal use only)
//	%^ int   	C_* classes (for liblink internal use)

#pragma	varargck	type	"$"	char*
#pragma	varargck	type	"M"	Addr*

void
listinit9(void)
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
}

static Prog*	bigP;

static int
Pconv(Fmt *fp)
{
	char str[STRINGSZ];
	Prog *p;
	int a;

	p = va_arg(fp->args, Prog*);
	bigP = p;
	a = p->as;

	str[0] = 0;
	if(a == ADATA)
		sprint(str, "%.5lld (%L)	%A	%D/%lld,%D", p->pc, p->lineno, a, &p->from, p->from3.offset, &p->to);
	else if(a == ATEXT || a == AGLOBL) {
		if(p->from3.offset != 0)
			sprint(str, "%.5lld (%L)	%A	%D,%lld,%D", p->pc, p->lineno, a, &p->from, p->from3.offset, &p->to);
		else
			sprint(str, "%.5lld (%L)	%A	%D,%D", p->pc, p->lineno, a, &p->from, &p->to);
	} else {
		if(p->mark & NOSCHED)
			sprint(strchr(str, 0), "*");
		if(p->reg == 0 && p->from3.type == TYPE_NONE)
			sprint(strchr(str, 0), "%.5lld (%L)	%A	%D,%D", p->pc, p->lineno, a, &p->from, &p->to);
		else
		if(a != ATEXT && p->from.type == TYPE_MEM) {
			sprint(strchr(str, 0), "%.5lld (%L)	%A	%lld(%R+%R),%D", p->pc, p->lineno, a,
				p->from.offset, p->from.reg, p->reg, &p->to);
		} else
		if(p->to.type == TYPE_MEM) {
			sprint(strchr(str, 0), "%.5lld (%L)	%A	%D,%lld(%R+%R)", p->pc, p->lineno, a,
					&p->from, p->to.offset, p->to.reg, p->reg);
		} else {
			sprint(strchr(str, 0), "%.5lld (%L)	%A	%D", p->pc, p->lineno, a, &p->from);
			if(p->reg != 0)
				sprint(strchr(str, 0), ",%R", p->reg);
			if(p->from3.type != TYPE_NONE)
				sprint(strchr(str, 0), ",%D", &p->from3);
			sprint(strchr(str, 0), ",%D", &p->to);
		}
		if(p->spadj != 0)
			return fmtprint(fp, "%s # spadj=%d", str, p->spadj);
	}
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
		s = anames9[a];
	return fmtstrcpy(fp, s);
}

static int
Dconv(Fmt *fp)
{
	char str[STRINGSZ];
	Addr *a;
	int32 v;

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
		if(bigP->pcond != nil) {
			v = bigP->pcond->pc;
			//if(v >= INITTEXT)
			//	v -= INITTEXT-HEADR;
			if(a->sym != nil)
				sprint(str, "%s+%.5ux(BRANCH)", a->sym->name, v);
			else
				sprint(str, "%.5ux(BRANCH)", v);
		} else if(a->u.branch != nil)
			sprint(str, "%lld", a->u.branch->pc);
		else if(a->sym != nil)
			sprint(str, "%s+%lld(APC)", a->sym->name, a->offset);
		else
			sprint(str, "%lld(APC)", a->offset);
		break;

	case TYPE_FCONST:
		//sprint(str, "$%lux-%lux", a->ieee.h, a->ieee.l);
		sprint(str, "$%.17g", a->u.dval);
		break;

	case TYPE_SCONST:
		sprint(str, "$\"%$\"", a->u.sval);
		break;
	}

	return fmtstrcpy(fp, str);
}

static int
Mconv(Fmt *fp)
{
	char str[STRINGSZ];
	Addr *a;
	LSym *s;
	int32 l;

	a = va_arg(fp->args, Addr*);
	s = a->sym;
	//if(s == nil) {
	//	l = a->offset;
	//	if((vlong)l != a->offset)
	//		sprint(str, "0x%llux", a->offset);
	//	else
	//		sprint(str, "%lld", a->offset);
	//	goto out;
	//}
	switch(a->name) {
	default:
		sprint(str, "GOK-name(%d)", a->name);
		break;

	case TYPE_NONE:
		l = a->offset;
		if((vlong)l != a->offset)
			sprint(str, "0x%llux", a->offset);
		else
			sprint(str, "%lld", a->offset);
		break;

	case NAME_EXTERN:
		if(a->offset != 0)
			sprint(str, "%s+%lld(SB)", s->name, a->offset);
		else
			sprint(str, "%s(SB)", s->name);
		break;

	case NAME_STATIC:
		sprint(str, "%s<>+%lld(SB)", s->name, a->offset);
		break;

	case NAME_AUTO:
		if(s == nil)
			sprint(str, "%lld(SP)", -a->offset);
		else
			sprint(str, "%s-%lld(SP)", s->name, -a->offset);
		break;

	case NAME_PARAM:
		if(s == nil)
			sprint(str, "%lld(FP)", a->offset);
		else
			sprint(str, "%s+%lld(FP)", s->name, a->offset);
		break;
	}
//out:
	return fmtstrcpy(fp, str);
}

static int
Rconv(Fmt *fp)
{
	int r;

	r = va_arg(fp->args, int);
	if(r == 0)
		return fmtstrcpy(fp, "NONE");
	if(REG_R0 <= r && r <= REG_R31)
		return fmtprint(fp, "R%d", r-REG_R0);
	if(REG_F0 <= r && r <= REG_F31)
		return fmtprint(fp, "F%d", r-REG_F0);
	if(REG_C0 <= r && r <= REG_C7)
		return fmtprint(fp, "C%d", r-REG_C0);
	if(r == REG_CR)
		return fmtstrcpy(fp, "CR");
	if(REG_SPR0 <= r && r <= REG_SPR0+1023) {
		switch(r) {
		case REG_XER:
			return fmtstrcpy(fp, "XER");
		case REG_LR:
			return fmtstrcpy(fp, "LR");
		case REG_CTR:
			return fmtstrcpy(fp, "CTR");
		}
		return fmtprint(fp, "SPR(%d)", r-REG_SPR0);
	}
	if(REG_DCR0 <= r && r <= REG_DCR0+1023)
		return fmtprint(fp, "DCR(%d)", r-REG_DCR0);
	if(r == REG_FPSCR)
		return fmtstrcpy(fp, "FPSCR");
	if(r == REG_MSR)
		return fmtstrcpy(fp, "MSR");

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
		s = cnames9[a];
	return fmtstrcpy(fp, s);
}

static int
DSconv(Fmt *fp)
{
	int i, c;
	char str[STRINGSZ], *p, *a;

	a = va_arg(fp->args, char*);
	p = str;
	for(i=0; i<sizeof(int32); i++) {
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
		}
		*p++ = (c>>6) + '0';
		*p++ = ((c>>3) & 7) + '0';
		*p++ = (c & 7) + '0';
	}
	*p = 0;
	return fmtstrcpy(fp, str);
}

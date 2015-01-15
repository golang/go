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
//		Flags: "%lD": seperate the high and low words of a constant by "-"
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
	int a, ch;

	p = va_arg(fp->args, Prog*);
	bigP = p;
	a = p->as;

	if(a == ADATA || a == AINIT || a == ADYNT)
		sprint(str, "%.5lld (%L)	%A	%D/%d,%D", p->pc, p->lineno, a, &p->from, p->reg, &p->to);
	else if(a == ATEXT) {
		if(p->reg != 0)
			sprint(str, "%.5lld (%L)        %A      %D,%d,%lD", p->pc, p->lineno, a, &p->from, p->reg, &p->to);
		else
			sprint(str, "%.5lld (%L)        %A      %D,%lD", p->pc, p->lineno, a, &p->from, &p->to);
	} else if(a == AGLOBL) {
		if(p->reg != 0)
			sprint(str, "%.5lld (%L)        %A      %D,%d,%D", p->pc, p->lineno, a, &p->from, p->reg, &p->to);
		else
			sprint(str, "%.5lld (%L)        %A      %D,%D", p->pc, p->lineno, a, &p->from, &p->to);
	} else {
		if(p->mark & NOSCHED)
			sprint(strchr(str, 0), "*");
		if(p->reg == NREG && p->from3.type == D_NONE)
			sprint(strchr(str, 0), "%.5lld (%L)	%A	%D,%D", p->pc, p->lineno, a, &p->from, &p->to);
		else
		if(a != ATEXT && p->from.type == D_OREG) {
			sprint(strchr(str, 0), "%.5lld (%L)	%A	%lld(R%d+R%d),%D", p->pc, p->lineno, a,
				p->from.offset, p->from.reg, p->reg, &p->to);
		} else
		if(p->to.type == D_OREG) {
			sprint(strchr(str, 0), "%.5lld (%L)	%A	%D,%lld(R%d+R%d)", p->pc, p->lineno, a,
					&p->from, p->to.offset, p->to.reg, p->reg);
		} else {
			sprint(strchr(str, 0), "%.5lld (%L)	%A	%D", p->pc, p->lineno, a, &p->from);
			if(p->reg != NREG) {
				ch = 'R';
				if(p->from.type == D_FREG)
					ch = 'F';
				sprint(strchr(str, 0), ",%c%d", ch, p->reg);
			}
			if(p->from3.type != D_NONE)
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

	if(fp->flags & FmtLong) {
		if(a->type == D_CONST)
			sprint(str, "$%d-%d", (int32)a->offset, (int32)(a->offset>>32));
		else {
			// ATEXT dst is not constant
			sprint(str, "!!%D", a);
		}
		goto ret;
	}

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
	case D_DCONST:
		if(a->reg != NREG)
			sprint(str, "$%M(R%d)", a, a->reg);
		else
			sprint(str, "$%M", a);
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
			sprint(str, "%M(F%d)(REG)", a, a->reg);
		break;

	case D_CREG:
		if(a->reg == NREG)
			strcpy(str, "CR");
		else
			sprint(str, "CR%d", a->reg);
		if(a->name != D_NONE || a->sym != nil)
			sprint(str, "%M(C%d)(REG)", a, a->reg);
		break;

	case D_SPR:
		if(a->name == D_NONE && a->sym == nil) {
			switch((ulong)a->offset) {
			case D_XER: sprint(str, "XER"); break;
			case D_LR: sprint(str, "LR"); break;
			case D_CTR: sprint(str, "CTR"); break;
			default: sprint(str, "SPR(%lld)", a->offset); break;
			}
			break;
		}
		sprint(str, "SPR-GOK(%d)", a->reg);
		if(a->name != D_NONE || a->sym != nil)
			sprint(str, "%M(SPR-GOK%d)(REG)", a, a->reg);
		break;

	case D_DCR:
		if(a->name == D_NONE && a->sym == nil) {
			sprint(str, "DCR(%lld)", a->offset);
			break;
		}
		sprint(str, "DCR-GOK(%d)", a->reg);
		if(a->name != D_NONE || a->sym != nil)
			sprint(str, "%M(DCR-GOK%d)(REG)", a, a->reg);
		break;

	case D_OPT:
		sprint(str, "OPT(%d)", a->reg);
		break;

	case D_FPSCR:
		if(a->reg == NREG)
			strcpy(str, "FPSCR");
		else
			sprint(str, "FPSCR(%d)", a->reg);
		break;

	case D_MSR:
		sprint(str, "MSR");
		break;

	case D_BRANCH:
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

	case D_FCONST:
		//sprint(str, "$%lux-%lux", a->ieee.h, a->ieee.l);
		sprint(str, "$%.17g", a->u.dval);
		break;

	case D_SCONST:
		sprint(str, "$\"%$\"", a->u.sval);
		break;
	}

ret:
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

	case D_NONE:
		l = a->offset;
		if((vlong)l != a->offset)
			sprint(str, "0x%llux", a->offset);
		else
			sprint(str, "%lld", a->offset);
		break;

	case D_EXTERN:
		if(a->offset != 0)
			sprint(str, "%s+%lld(SB)", s->name, a->offset);
		else
			sprint(str, "%s(SB)", s->name);
		break;

	case D_STATIC:
		sprint(str, "%s<>+%lld(SB)", s->name, a->offset);
		break;

	case D_AUTO:
		if(s == nil)
			sprint(str, "%lld(SP)", -a->offset);
		else
			sprint(str, "%s-%lld(SP)", s->name, -a->offset);
		break;

	case D_PARAM:
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
	char str[STRINGSZ];
	int r;

	r = va_arg(fp->args, int);
	if(r < NREG)
		sprint(str, "r%d", r);
	else
		sprint(str, "f%d", r-NREG);
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

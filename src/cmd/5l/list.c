// Inferno utils/5l/list.h
// http://code.google.com/p/inferno-os/source/browse/utils/5l/list.c
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

#include "l.h"
#include "../ld/lib.h"

void
listinit(void)
{

	fmtinstall('A', Aconv);
	fmtinstall('C', Cconv);
	fmtinstall('D', Dconv);
	fmtinstall('P', Pconv);
	fmtinstall('S', Sconv);
	fmtinstall('N', Nconv);
	fmtinstall('O', Oconv);		// C_type constants
	fmtinstall('I', Iconv);
}

void
prasm(Prog *p)
{
	print("%P\n", p);
}

int
Pconv(Fmt *fp)
{
	Prog *p;
	int a;

	p = va_arg(fp->args, Prog*);
	curp = p;
	a = p->as;
	switch(a) {
	default:
		fmtprint(fp, "(%d)", p->line);
		if(p->reg == NREG && p->as != AGLOBL)
			fmtprint(fp, "	%A%C	%D,%D",
				a, p->scond, &p->from, &p->to);
		else
		if(p->from.type != D_FREG)
			fmtprint(fp, "	%A%C	%D,R%d,%D",
				a, p->scond, &p->from, p->reg, &p->to);
		else
			fmtprint(fp, "	%A%C	%D,F%d,%D",
				a, p->scond, &p->from, p->reg, &p->to);
		break;

	case ASWPW:
	case ASWPBU:
		fmtprint(fp, "(%d)	%A%C	R%d,%D,%D",
			p->line, a, p->scond, p->reg, &p->from, &p->to);
		break;

	case ADATA:
	case AINIT_:
	case ADYNT_:
		fmtprint(fp, "(%d)	%A%C	%D/%d,%D",
			p->line, a, p->scond, &p->from, p->reg, &p->to);
		break;

	case AWORD:
		fmtprint(fp, "(%d)	WORD	%D", p->line, &p->to);
		break;

	case ADWORD:
		fmtprint(fp, "(%d)	DWORD	%D %D", p->line, &p->from, &p->to);
		break;
	}
	
	if(p->spadj)
		fmtprint(fp, "  (spadj%+d)", p->spadj);

	return 0;
}

int
Aconv(Fmt *fp)
{
	char *s;
	int a;

	a = va_arg(fp->args, int);
	s = "???";
	if(a >= AXXX && a < ALAST)
		s = anames[a];
	return fmtstrcpy(fp, s);
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
	char s[20];
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
Dconv(Fmt *fp)
{
	char str[STRINGSZ];
	const char *op;
	Adr *a;
	int32 v;

	a = va_arg(fp->args, Adr*);
	switch(a->type) {

	default:
		snprint(str, sizeof str, "GOK-type(%d)", a->type);
		break;

	case D_NONE:
		str[0] = 0;
		if(a->name != D_NONE || a->reg != NREG || a->sym != S)
			snprint(str, sizeof str, "%N(R%d)(NONE)", a, a->reg);
		break;

	case D_CONST:
		if(a->reg == NREG)
			snprint(str, sizeof str, "$%N", a);
		else
			snprint(str, sizeof str, "$%N(R%d)", a, a->reg);
		break;

	case D_CONST2:
		snprint(str, sizeof str, "$%d-%d", a->offset, a->offset2);
		break;

	case D_SHIFT:
		v = a->offset;
		op = &"<<>>->@>"[(((v>>5) & 3) << 1)];
		if(v & (1<<4))
			snprint(str, sizeof str, "R%d%c%cR%d", v&15, op[0], op[1], (v>>8)&15);
		else
			snprint(str, sizeof str, "R%d%c%c%d", v&15, op[0], op[1], (v>>7)&31);
		if(a->reg != NREG)
			seprint(str+strlen(str), str+sizeof str, "(R%d)", a->reg);
		break;

	case D_OCONST:
		snprint(str, sizeof str, "$*$%N", a);
		if(a->reg != NREG)
			snprint(str, sizeof str, "%N(R%d)(CONST)", a, a->reg);
		break;

	case D_OREG:
		if(a->reg != NREG)
			snprint(str, sizeof str, "%N(R%d)", a, a->reg);
		else
			snprint(str, sizeof str, "%N", a);
		break;

	case D_REG:
		snprint(str, sizeof str, "R%d", a->reg);
		if(a->name != D_NONE || a->sym != S)
			snprint(str, sizeof str, "%N(R%d)(REG)", a, a->reg);
		break;

	case D_REGREG:
		snprint(str, sizeof str, "(R%d,R%d)", a->reg, (int)a->offset);
		if(a->name != D_NONE || a->sym != S)
			snprint(str, sizeof str, "%N(R%d)(REG)", a, a->reg);
		break;

	case D_REGREG2:
		snprint(str, sizeof str, "R%d,R%d", a->reg, (int)a->offset);
		if(a->name != D_NONE || a->sym != S)
			snprint(str, sizeof str, "%N(R%d)(REG)", a, a->reg);
		break;

	case D_FREG:
		snprint(str, sizeof str, "F%d", a->reg);
		if(a->name != D_NONE || a->sym != S)
			snprint(str, sizeof str, "%N(R%d)(REG)", a, a->reg);
		break;

	case D_PSR:
		switch(a->reg) {
		case 0:
			snprint(str, sizeof str, "CPSR");
			break;
		case 1:
			snprint(str, sizeof str, "SPSR");
			break;
		default:
			snprint(str, sizeof str, "PSR%d", a->reg);
			break;
		}
		if(a->name != D_NONE || a->sym != S)
			snprint(str, sizeof str, "%N(PSR%d)(REG)", a, a->reg);
		break;

	case D_FPCR:
		switch(a->reg){
		case 0:
			snprint(str, sizeof str, "FPSR");
			break;
		case 1:
			snprint(str, sizeof str, "FPCR");
			break;
		default:
			snprint(str, sizeof str, "FCR%d", a->reg);
			break;
		}
		if(a->name != D_NONE || a->sym != S)
			snprint(str, sizeof str, "%N(FCR%d)(REG)", a, a->reg);

		break;

	case D_BRANCH:	/* botch */
		if(curp->cond != P) {
			v = curp->cond->pc;
			if(a->sym != S)
				snprint(str, sizeof str, "%s+%.5ux(BRANCH)", a->sym->name, v);
			else
				snprint(str, sizeof str, "%.5ux(BRANCH)", v);
		} else
			if(a->sym != S)
				snprint(str, sizeof str, "%s+%d(APC)", a->sym->name, a->offset);
			else
				snprint(str, sizeof str, "%d(APC)", a->offset);
		break;

	case D_FCONST:
		snprint(str, sizeof str, "$%e", ieeedtod(&a->ieee));
		break;

	case D_SCONST:
		snprint(str, sizeof str, "$\"%S\"", a->sval);
		break;
	}
	return fmtstrcpy(fp, str);
}

int
Nconv(Fmt *fp)
{
	char str[STRINGSZ];
	Adr *a;
	Sym *s;

	a = va_arg(fp->args, Adr*);
	s = a->sym;
	switch(a->name) {
	default:
		sprint(str, "GOK-name(%d)", a->name);
		break;

	case D_NONE:
		sprint(str, "%d", a->offset);
		break;

	case D_EXTERN:
		if(s == S)
			sprint(str, "%d(SB)", a->offset);
		else
			sprint(str, "%s+%d(SB)", s->name, a->offset);
		break;

	case D_STATIC:
		if(s == S)
			sprint(str, "<>+%d(SB)", a->offset);
		else
			sprint(str, "%s<>+%d(SB)", s->name, a->offset);
		break;

	case D_AUTO:
		if(s == S)
			sprint(str, "%d(SP)", a->offset);
		else
			sprint(str, "%s-%d(SP)", s->name, -a->offset);
		break;

	case D_PARAM:
		if(s == S)
			sprint(str, "%d(FP)", a->offset);
		else
			sprint(str, "%s+%d(FP)", s->name, a->offset);
		break;
	}
	return fmtstrcpy(fp, str);
}

int
Sconv(Fmt *fp)
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

int
Iconv(Fmt *fp)
{
	int i, n;
	uint32 *p;
	char *s;
	Fmt fmt;
	
	n = fp->prec;
	fp->prec = 0;
	if(!(fp->flags&FmtPrec) || n < 0)
		return fmtstrcpy(fp, "%I");
	fp->flags &= ~FmtPrec;
	p = va_arg(fp->args, uint32*);

	// format into temporary buffer and
	// call fmtstrcpy to handle padding.
	fmtstrinit(&fmt);
	for(i=0; i<n/4; i++) {
		if(i > 0)
			fmtprint(&fmt, " ");
		fmtprint(&fmt, "%.8ux", *p++);
	}
	s = fmtstrflush(&fmt);
	fmtstrcpy(fp, s);
	free(s);
	return 0;
}

static char*
cnames[] =
{
	[C_ADDR]	= "C_ADDR",
	[C_FAUTO]	= "C_FAUTO",
	[C_ZFCON]	= "C_SFCON",
	[C_SFCON]	= "C_SFCON",
	[C_LFCON]	= "C_LFCON",
	[C_FCR]		= "C_FCR",
	[C_FOREG]	= "C_FOREG",
	[C_FREG]	= "C_FREG",
	[C_GOK]		= "C_GOK",
	[C_HAUTO]	= "C_HAUTO",
	[C_HFAUTO]	= "C_HFAUTO",
	[C_HFOREG]	= "C_HFOREG",
	[C_HOREG]	= "C_HOREG",
	[C_HREG]	= "C_HREG",
	[C_LACON]	= "C_LACON",
	[C_LAUTO]	= "C_LAUTO",
	[C_LBRA]	= "C_LBRA",
	[C_LCON]	= "C_LCON",
	[C_LCONADDR]	= "C_LCONADDR",
	[C_LOREG]	= "C_LOREG",
	[C_NCON]	= "C_NCON",
	[C_NONE]	= "C_NONE",
	[C_PC]		= "C_PC",
	[C_PSR]		= "C_PSR",
	[C_RACON]	= "C_RACON",
	[C_RCON]	= "C_RCON",
	[C_REG]		= "C_REG",
	[C_REGREG]	= "C_REGREG",
	[C_REGREG2]	= "C_REGREG2",
	[C_ROREG]	= "C_ROREG",
	[C_SAUTO]	= "C_SAUTO",
	[C_SBRA]	= "C_SBRA",
	[C_SCON]	= "C_SCON",
	[C_SHIFT]	= "C_SHIFT",
	[C_SOREG]	= "C_SOREG",
	[C_SP]		= "C_SP",
	[C_SROREG]	= "C_SROREG"
};

int
Oconv(Fmt *fp)
{
	char buf[500];
	int o;

	o = va_arg(fp->args, int);
	if(o < 0 || o >= nelem(cnames) || cnames[o] == nil) {
		snprint(buf, sizeof(buf), "C_%d", o);
		return fmtstrcpy(fp, buf);
	}
	return fmtstrcpy(fp, cnames[o]);
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

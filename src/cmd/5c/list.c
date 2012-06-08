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


#define	EXTERN
#include "gc.h"

void
listinit(void)
{

	fmtinstall('A', Aconv);
	fmtinstall('P', Pconv);
	fmtinstall('S', Sconv);
	fmtinstall('N', Nconv);
	fmtinstall('B', Bconv);
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
			sprint(ss, "$%d", var[i].offset);
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

char *extra [] = {
	".EQ", ".NE", ".CS", ".CC",
	".MI", ".PL", ".VS", ".VC",
	".HI", ".LS", ".GE", ".LT",
	".GT", ".LE", "", ".NV",
};

int
Pconv(Fmt *fp)
{
	char str[STRINGSZ], sc[20];
	Prog *p;
	int a, s;

	p = va_arg(fp->args, Prog*);
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
			sprint(str, "	%A%s	%R,%D", a, sc, &p->from, &p->to);
		else
		if(p->to.type == D_CONST)
			sprint(str, "	%A%s	%D,%R", a, sc, &p->from, &p->to);
		else
			sprint(str, "	%A%s	%D,%D", a, sc, &p->from, &p->to);
	} else
	if(a == ADATA)
		sprint(str, "	%A	%D/%d,%D", a, &p->from, p->reg, &p->to);
	else
	if(p->as == ATEXT)
		sprint(str, "	%A	%D,%d,%D", a, &p->from, p->reg, &p->to);
	else
	if(p->reg == NREG)
		sprint(str, "	%A%s	%D,%D", a, sc, &p->from, &p->to);
	else
	if(p->from.type != D_FREG)
		sprint(str, "	%A%s	%D,R%d,%D", a, sc, &p->from, p->reg, &p->to);
	else
		sprint(str, "	%A%s	%D,F%d,%D", a, sc, &p->from, p->reg, &p->to);
	return fmtstrcpy(fp, str);
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

int
Dconv(Fmt *fp)
{
	char str[STRINGSZ];
	Adr *a;
	const char *op;
	int v;

	a = va_arg(fp->args, Adr*);
	switch(a->type) {

	default:
		sprint(str, "GOK-type(%d)", a->type);
		break;

	case D_NONE:
		str[0] = 0;
		if(a->name != D_NONE || a->reg != NREG || a->sym != S)
			sprint(str, "%N(R%d)(NONE)", a, a->reg);
		break;

	case D_CONST:
		if(a->reg != NREG)
			sprint(str, "$%N(R%d)", a, a->reg);
		else
			sprint(str, "$%N", a);
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

	case D_OREG:
		if(a->reg != NREG)
			sprint(str, "%N(R%d)", a, a->reg);
		else
			sprint(str, "%N", a);
		break;

	case D_REG:
		sprint(str, "R%d", a->reg);
		if(a->name != D_NONE || a->sym != S)
			sprint(str, "%N(R%d)(REG)", a, a->reg);
		break;

	case D_FREG:
		sprint(str, "F%d", a->reg);
		if(a->name != D_NONE || a->sym != S)
			sprint(str, "%N(R%d)(REG)", a, a->reg);
		break;

	case D_PSR:
		sprint(str, "PSR");
		if(a->name != D_NONE || a->sym != S)
			sprint(str, "%N(PSR)(REG)", a);
		break;

	case D_BRANCH:
		sprint(str, "%d(PC)", a->offset-pc);
		break;

	case D_FCONST:
		sprint(str, "$%.17e", a->dval);
		break;

	case D_SCONST:
		sprint(str, "$\"%S\"", a->sval);
		break;
	}
	return fmtstrcpy(fp, str);
}

int
Rconv(Fmt *fp)
{
	char str[STRINGSZ];
	Adr *a;
	int i, v;

	a = va_arg(fp->args, Adr*);
	sprint(str, "GOK-reglist");
	switch(a->type) {
	case D_CONST:
	case D_CONST2:
		if(a->reg != NREG)
			break;
		if(a->sym != S)
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

int
Sconv(Fmt *fp)
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

int
Nconv(Fmt *fp)
{
	char str[STRINGSZ];
	Adr *a;
	Sym *s;

	a = va_arg(fp->args, Adr*);
	s = a->sym;
	if(s == S) {
		sprint(str, "%d", a->offset);
		goto out;
	}
	switch(a->name) {
	default:
		sprint(str, "GOK-name(%d)", a->name);
		break;

	case D_NONE:
		sprint(str, "%d", a->offset);
		break;

	case D_EXTERN:
		sprint(str, "%s+%d(SB)", s->name, a->offset);
		break;

	case D_STATIC:
		sprint(str, "%s<>+%d(SB)", s->name, a->offset);
		break;

	case D_AUTO:
		sprint(str, "%s-%d(SP)", s->name, -a->offset);
		break;

	case D_PARAM:
		sprint(str, "%s+%d(FP)", s->name, a->offset);
		break;
	}
out:
	return fmtstrcpy(fp, str);
}

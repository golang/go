// Inferno utils/5l/obj.c
// http://code.google.com/p/inferno-os/source/browse/utils/5l/obj.c
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
#include	"l.h"
#include	"../ld/lib.h"
#include	<ar.h>

#ifndef	DEFAULT
#define	DEFAULT	'9'
#endif

char	*noname		= "<none>";
char	thechar		= '5';
char	*thestring 	= "arm";

/*
 *	-H1 -T0x10005000 -R4		is aif for risc os
 *	-H2 -T4128 -R4096		is plan9 format
 *	-H3 -T0xF0000020 -R4		is NetBSD format
 *	-H4				is IXP1200 (raw)
 *	-H5 -T0xC0008010 -R1024		is ipaq
 */

static int
isobjfile(char *f)
{
	int n, v;
	Biobuf *b;
	char buf1[5], buf2[SARMAG];

	b = Bopen(f, OREAD);
	if(b == nil)
		return 0;
	n = Bread(b, buf1, 5);
	if(n == 5 && (buf1[2] == 1 && buf1[3] == '<' || buf1[3] == 1 && buf1[4] == '<'))
		v = 1;	/* good enough for our purposes */
	else{
		Bseek(b, 0, 0);
		n = Bread(b, buf2, SARMAG);
		v = n == SARMAG && strncmp(buf2, ARMAG, SARMAG) == 0;
	}
	Bterm(b);
	return v;
}

static char*
linkername[] =
{
	"runtime·softfloat",
};

void
usage(void)
{
	fprint(2, "usage: 5l [-options] objects\n");
	errorexit();
}

void
main(int argc, char *argv[])
{
	int c, i;

	Binit(&bso, 1, OWRITE);
	cout = -1;
	listinit();
	nerrors = 0;
	curtext = P;
	outfile = "5.out";
	HEADTYPE = -1;
	INITTEXT = -1;
	INITDAT = -1;
	INITRND = -1;
	INITENTRY = 0;

	ARGBEGIN {
	default:
		c = ARGC();
		if(c >= 0 && c < sizeof(debug))
			debug[c]++;
		break;
	case 'o':
		outfile = EARGF(usage());
		break;
	case 'E':
		INITENTRY = EARGF(usage());
		break;
	case 'L':
		Lflag(EARGF(usage()));
		break;
	case 'T':
		INITTEXT = atolwhex(EARGF(usage()));
		break;
	case 'D':
		INITDAT = atolwhex(EARGF(usage()));
		break;
	case 'R':
		INITRND = atolwhex(EARGF(usage()));
		break;
	case 'H':
		HEADTYPE = atolwhex(EARGF(usage()));
		/* do something about setting INITTEXT */
		break;
	case 'x':	/* produce export table */
		doexp = 1;
		if(argv[1] != nil && argv[1][0] != '-' && !isobjfile(argv[1]))
			readundefs(ARGF(), SEXPORT);
		break;
	case 'u':	/* produce dynamically loadable module */
		dlm = 1;
		debug['l']++;
		if(argv[1] != nil && argv[1][0] != '-' && !isobjfile(argv[1]))
			readundefs(ARGF(), SIMPORT);
		break;
	} ARGEND

	USED(argc);

	if(*argv == 0)
		usage();

	libinit();

	if(!debug['9'] && !debug['U'] && !debug['B'])
		debug[DEFAULT] = 1;
	if(HEADTYPE == -1) {
		if(debug['U'])
			HEADTYPE = 0;
		if(debug['B'])
			HEADTYPE = 1;
		if(debug['9'])
			HEADTYPE = 2;
		HEADTYPE = 6;
	}
	switch(HEADTYPE) {
	default:
		diag("unknown -H option");
		errorexit();
	case 0:	/* no header */
		HEADR = 0L;
		if(INITTEXT == -1)
			INITTEXT = 0;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 4;
		break;
	case 1:	/* aif for risc os */
		HEADR = 128L;
		if(INITTEXT == -1)
			INITTEXT = 0x10005000 + HEADR;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 4;
		break;
	case 2:	/* plan 9 */
		HEADR = 32L;
		if(INITTEXT == -1)
			INITTEXT = 4128;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 4096;
		break;
	case 3:	/* boot for NetBSD */
		HEADR = 32L;
		if(INITTEXT == -1)
			INITTEXT = 0xF0000020L;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 4096;
		break;
	case 4: /* boot for IXP1200 */
		HEADR = 0L;
		if(INITTEXT == -1)
			INITTEXT = 0x0;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 4;
		break;
	case 5: /* boot for ipaq */
		HEADR = 16L;
		if(INITTEXT == -1)
			INITTEXT = 0xC0008010;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 1024;
		break;
	case 6:	/* arm elf */
		HEADR = linuxheadr();
		if(INITTEXT == -1)
			INITTEXT = 0x8000+HEADR;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 4096;
		break;
	}
	if(INITDAT != 0 && INITRND != 0)
		print("warning: -D0x%lux is ignored because of -R0x%lux\n",
			INITDAT, INITRND);
	if(debug['v'])
		Bprint(&bso, "HEADER = -H0x%d -T0x%lux -D0x%lux -R0x%lux\n",
			HEADTYPE, INITTEXT, INITDAT, INITRND);
	Bflush(&bso);
	zprg.as = AGOK;
	zprg.scond = 14;
	zprg.reg = NREG;
	zprg.from.name = D_NONE;
	zprg.from.type = D_NONE;
	zprg.from.reg = NREG;
	zprg.to = zprg.from;
	buildop();
	thumbbuildop();	// could build on demand
	histgen = 0;
	textp = P;
	datap = P;
	edatap = P;
	pc = 0;
	dtype = 4;
	nuxiinit();

	version = 0;
	cbp = buf.cbuf;
	cbc = sizeof(buf.cbuf);
	firstp = prg();
	lastp = firstp;

	while(*argv)
		objfile(*argv++);
	if(!debug['l'])
		loadlib();

	// mark some functions that are only referenced after linker code editing
	// TODO(kaib): this doesn't work, the prog can't be found in runtime
	for(i=0; i<nelem(linkername); i++)
		mark(lookup(linkername[i], 0));
	deadcode();

	firstp = firstp->link;
	if(firstp == P)
		goto out;
	if(doexp || dlm){
		EXPTAB = "_exporttab";
		zerosig(EXPTAB);
		zerosig("etext");
		zerosig("edata");
		zerosig("end");
		if(dlm){
			initdiv();
			import();
			HEADTYPE = 2;
			INITTEXT = INITDAT = 0;
			INITRND = 8;
			INITENTRY = EXPTAB;
		}
		else
			divsig();
		export();
	}
	patch();
	if(debug['p'])
		if(debug['1'])
			doprof1();
		else
			doprof2();
	if(debug['u'])
		reachable();
	dodata();
	if(seenthumb && debug['f'])
		fnptrs();
	follow();
	if(firstp == P)
		goto out;
	softfloat();
	noops();
	span();
	asmb();
	undef();

out:
	if(debug['c']){
		thumbcount();
		print("ARM size = %d\n", armsize);
	}
	if(debug['v']) {
		Bprint(&bso, "%5.2f cpu time\n", cputime());
		Bprint(&bso, "%d sizeof adr\n", sizeof(Adr));
		Bprint(&bso, "%d sizeof prog\n", sizeof(Prog));
	}
	Bflush(&bso);
	errorexit();
}

void
zaddr(Biobuf *f, Adr *a, Sym *h[])
{
	int i, c;
	int32 l;
	Sym *s;
	Auto *u;

	a->type = Bgetc(f);
	a->reg = Bgetc(f);
	c = Bgetc(f);
	if(c < 0 || c > NSYM){
		print("sym out of range: %d\n", c);
		Bputc(f, ALAST+1);
		return;
	}
	a->sym = h[c];
	a->name = Bgetc(f);

	if(a->reg < 0 || a->reg > NREG) {
		print("register out of range %d\n", a->reg);
		Bputc(f, ALAST+1);
		return;	/*  force real diagnostic */
	}

	if(a->type == D_CONST || a->type == D_OCONST) {
		if(a->name == D_EXTERN || a->name == D_STATIC) {
			s = a->sym;
			if(s != S && (s->type == STEXT || s->type == SLEAF || s->type == SCONST || s->type == SXREF)) {
				if(0 && !s->fnptr && s->name[0] != '.')
					print("%s used as function pointer\n", s->name);
				s->fnptr = 1;	// over the top cos of SXREF
			}
		}
	}

	switch(a->type) {
	default:
		print("unknown type %d\n", a->type);
		Bputc(f, ALAST+1);
		return;	/*  force real diagnostic */

	case D_NONE:
	case D_REG:
	case D_FREG:
	case D_PSR:
	case D_FPCR:
		break;

	case D_REGREG:
		a->offset = Bgetc(f);
		c++;
		break;

	case D_CONST2:
		a->offset2 = Bget4(f);	// fall through
	case D_BRANCH:
	case D_OREG:
	case D_CONST:
	case D_OCONST:
	case D_SHIFT:
		a->offset = Bget4(f);
		break;

	case D_SCONST:
		a->sval = mal(NSNAME);
		Bread(f, a->sval, NSNAME);
		c += NSNAME;
		break;

	case D_FCONST:
		a->ieee = mal(sizeof(Ieee));
		a->ieee->l = Bget4(f);
		a->ieee->h = Bget4(f);
		break;
	}
	s = a->sym;
	if(s == S)
		return;
	i = a->name;
	if(i != D_AUTO && i != D_PARAM)
		return;

	l = a->offset;
	for(u=curauto; u; u=u->link)
		if(u->asym == s)
		if(u->type == i) {
			if(u->aoffset > l)
				u->aoffset = l;
			return;
		}

	u = mal(sizeof(Auto));
	u->link = curauto;
	curauto = u;
	u->asym = s;
	u->aoffset = l;
	u->type = i;
}

void
nopout(Prog *p)
{
	p->as = ANOP;
	p->from.type = D_NONE;
	p->to.type = D_NONE;
}

static void puntfp(Prog *);

void
ldobj1(Biobuf *f, int64 len, char *pn)
{
	int32 ipc;
	Prog *p, *t;
	Sym *h[NSYM], *s, *di;
	int v, o, r, skip;
	uint32 sig;
	char *name;
	int ntext;
	int32 eof;
	char src[1024];

	ntext = 0;
	eof = Boffset(f) + len;
	di = S;
	src[0] = 0;

newloop:
	memset(h, 0, sizeof(h));
	version++;
	histfrogp = 0;
	ipc = pc;
	skip = 0;

loop:
	if(f->state == Bracteof || Boffset(f) >= eof)
		goto eof;
	o = Bgetc(f);
	if(o == Beof)
		goto eof;

	if(o <= AXXX || o >= ALAST) {
		diag("%s:#%lld: opcode out of range: %#ux", pn, Boffset(f), o);
		print("	probably not a .5 file\n");
		errorexit();
	}
	if(o == ANAME || o == ASIGNAME) {
		sig = 0;
		if(o == ASIGNAME)
			sig = Bget4(f);
		v = Bgetc(f); /* type */
		o = Bgetc(f); /* sym */
		r = 0;
		if(v == D_STATIC)
			r = version;
		name = Brdline(f, '\0');
		if(name == nil) {
			if(Blinelen(f) > 0) {
				fprint(2, "%s: name too long\n", pn);
				errorexit();
			}
			goto eof;
		}
		s = lookup(name, r);

		if(sig != 0){
			if(s->sig != 0 && s->sig != sig)
				diag("incompatible type signatures %lux(%s) and %lux(%s) for %s", s->sig, s->file, sig, pn, s->name);
			s->sig = sig;
			s->file = pn;
		}

		if(debug['W'])
			print("	ANAME	%s\n", s->name);
		h[o] = s;
		if((v == D_EXTERN || v == D_STATIC) && s->type == 0)
			s->type = SXREF;
		if(v == D_FILE) {
			if(s->type != SFILE) {
				histgen++;
				s->type = SFILE;
				s->value = histgen;
			}
			if(histfrogp < MAXHIST) {
				histfrog[histfrogp] = s;
				histfrogp++;
			} else
				collapsefrog(s);
		}
		goto loop;
	}

	p = mal(sizeof(Prog));
	p->as = o;
	p->scond = Bgetc(f);
	p->reg = Bgetc(f);
	p->line = Bget4(f);

	zaddr(f, &p->from, h);
	zaddr(f, &p->to, h);

	if(p->reg > NREG)
		diag("register out of range %d", p->reg);

	p->link = P;
	p->cond = P;

	if(debug['W'])
		print("%P\n", p);

	switch(o) {
	case AHISTORY:
		if(p->to.offset == -1) {
			addlib(src, pn);
			histfrogp = 0;
			goto loop;
		}
		if(src[0] == '\0')
			copyhistfrog(src, sizeof src);
		addhist(p->line, D_FILE);		/* 'z' */
		if(p->to.offset)
			addhist(p->to.offset, D_FILE1);	/* 'Z' */
		histfrogp = 0;
		goto loop;

	case AEND:
		histtoauto();
		if(curtext != P)
			curtext->to.autom = curauto;
		curauto = 0;
		curtext = P;
		if(Boffset(f) == eof)
			return;
		goto newloop;

	case AGLOBL:
		s = p->from.sym;
		if(s == S) {
			diag("GLOBL must have a name\n%P", p);
			errorexit();
		}
		if(s->type == 0 || s->type == SXREF) {
			s->type = SBSS;
			s->value = 0;
		}
		if(s->type != SBSS) {
			diag("redefinition: %s\n%P", s->name, p);
			s->type = SBSS;
			s->value = 0;
		}
		if(p->to.offset > s->value)
			s->value = p->to.offset;
		if(p->reg & DUPOK)
			s->dupok = 1;
		break;

	case ADYNT:
		s = p->from.sym;
		if(p->to.sym == S) {
			diag("DYNT without a sym\n%P", p);
			break;
		}
		di = p->to.sym;
		p->reg = 4;
		if(di->type == SXREF) {
			if(debug['z'])
				Bprint(&bso, "%P set to %d\n", p, dtype);
			di->type = SCONST;
			di->value = dtype;
			dtype += 4;
		}
		if(s == S)
			break;

		p->from.offset = di->value;
		s->type = SDATA;
		if(curtext == P) {
			diag("DYNT not in text: %P", p);
			break;
		}
		p->to.sym = curtext->from.sym;
		p->to.type = D_CONST;
		if(s != S) {
			p->dlink = s->data;
			s->data = p;
		}
		if(edatap == P)
			datap = p;
		else
			edatap->link = p;
		edatap = p;
		break;

	case AINIT:
		s = p->from.sym;
		if(s == S) {
			diag("INIT without a sym\n%P", p);
			break;
		}
		if(di == S) {
			diag("INIT without previous DYNT\n%P", p);
			break;
		}
		p->from.offset = di->value;
		s->type = SDATA;
		if(s != S) {
			p->dlink = s->data;
			s->data = p;
		}
		if(edatap == P)
			datap = p;
		else
			edatap->link = p;
		edatap = p;
		break;

	case ADATA:
		// Assume that AGLOBL comes after ADATA.
		// If we've seen an AGLOBL that said this sym was DUPOK,
		// ignore any more ADATA we see, which must be
		// redefinitions.
		s = p->from.sym;
		if(s != S && s->dupok) {
			if(debug['v'])
				Bprint(&bso, "skipping %s in %s: dupok\n", s->name, pn);
			goto loop;
		}
		if(s != S) {
			p->dlink = s->data;
			s->data = p;
		}
		if(edatap == P)
			datap = p;
		else
			edatap->link = p;
		edatap = p;
		break;

	case AGOK:
		diag("unknown opcode\n%P", p);
		p->pc = pc;
		pc++;
		break;

	case ATEXT:
		s = p->from.sym;
		if(ntext++ == 0 && s->type != 0 && s->type != SXREF) {
			/* redefinition, so file has probably been seen before */
			if(debug['v'])
				Bprint(&bso, "skipping: %s: redefinition: %s", pn, s->name);
			return;
		}
		setarch(p);
		setthumb(p);
		p->align = 4;
		if(curtext != P) {
			histtoauto();
			curtext->to.autom = curauto;
			curauto = 0;
		}
		skip = 0;
		curtext = p;
		autosize = (p->to.offset+3L) & ~3L;
		p->to.offset = autosize;
		autosize += 4;
		s = p->from.sym;
		if(s == S) {
			diag("TEXT must have a name\n%P", p);
			errorexit();
		}
		if(s->type != 0 && s->type != SXREF) {
			if(p->reg & DUPOK) {
				skip = 1;
				goto casedef;
			}
			diag("redefinition: %s\n%P", s->name, p);
		}
		s->type = STEXT;
		s->text = p;
		s->value = pc;
		s->thumb = thumb;
		lastp->link = p;
		lastp = p;
		p->pc = pc;
		pc++;
		if(textp == P) {
			textp = p;
			etextp = p;
			goto loop;
		}
		etextp->cond = p;
		etextp = p;
		break;

	case ASUB:
		if(p->from.type == D_CONST)
		if(p->from.name == D_NONE)
		if(p->from.offset < 0) {
			p->from.offset = -p->from.offset;
			p->as = AADD;
		}
		goto casedef;

	case AADD:
		if(p->from.type == D_CONST)
		if(p->from.name == D_NONE)
		if(p->from.offset < 0) {
			p->from.offset = -p->from.offset;
			p->as = ASUB;
		}
		goto casedef;

	case AMOVWD:
	case AMOVWF:
	case AMOVDW:
	case AMOVFW:
	case AMOVFD:
	case AMOVDF:
	// case AMOVF:
	// case AMOVD:
	case ACMPF:
	case ACMPD:
	case AADDF:
	case AADDD:
	case ASUBF:
	case ASUBD:
	case AMULF:
	case AMULD:
	case ADIVF:
	case ADIVD:
		if(thumb)
			puntfp(p);
		goto casedef;

	case AMOVF:
		if(thumb)
			puntfp(p);
		if(skip)
			goto casedef;

		if(p->from.type == D_FCONST && chipfloat(p->from.ieee) < 0) {
			/* size sb 9 max */
			sprint(literal, "$%lux", ieeedtof(p->from.ieee));
			s = lookup(literal, 0);
			if(s->type == 0) {
				s->type = SBSS;
				s->value = 4;
				t = prg();
				t->as = ADATA;
				t->line = p->line;
				t->from.type = D_OREG;
				t->from.sym = s;
				t->from.name = D_EXTERN;
				t->reg = 4;
				t->to = p->from;
				if(edatap == P)
					datap = t;
				else
					edatap->link = t;
				edatap = t;
				t->link = P;
			}
			p->from.type = D_OREG;
			p->from.sym = s;
			p->from.name = D_EXTERN;
			p->from.offset = 0;
		}
		goto casedef;

	case AMOVD:
		if(thumb)
			puntfp(p);
		if(skip)
			goto casedef;

		if(p->from.type == D_FCONST && chipfloat(p->from.ieee) < 0) {
			/* size sb 18 max */
			sprint(literal, "$%lux.%lux",
				p->from.ieee->l, p->from.ieee->h);
			s = lookup(literal, 0);
			if(s->type == 0) {
				s->type = SBSS;
				s->value = 8;
				t = prg();
				t->as = ADATA;
				t->line = p->line;
				t->from.type = D_OREG;
				t->from.sym = s;
				t->from.name = D_EXTERN;
				t->reg = 8;
				t->to = p->from;
				if(edatap == P)
					datap = t;
				else
					edatap->link = t;
				edatap = t;
				t->link = P;
			}
			p->from.type = D_OREG;
			p->from.sym = s;
			p->from.name = D_EXTERN;
			p->from.offset = 0;
		}
		goto casedef;

	default:
	casedef:
		if(skip)
			nopout(p);

		if(p->to.type == D_BRANCH)
			p->to.offset += ipc;
		lastp->link = p;
		lastp = p;
		p->pc = pc;
		pc++;
		break;
	}
	goto loop;

eof:
	diag("truncated object file: %s", pn);
}

Prog*
prg(void)
{
	Prog *p;

	p = mal(sizeof(Prog));
	*p = zprg;
	return p;
}

void
doprof1(void)
{
	Sym *s;
	int32 n;
	Prog *p, *q;

	if(debug['v'])
		Bprint(&bso, "%5.2f profile 1\n", cputime());
	Bflush(&bso);
	s = lookup("__mcount", 0);
	n = 1;
	for(p = firstp->link; p != P; p = p->link) {
		setarch(p);
		if(p->as == ATEXT) {
			q = prg();
			q->line = p->line;
			q->link = datap;
			datap = q;
			q->as = ADATA;
			q->from.type = D_OREG;
			q->from.name = D_EXTERN;
			q->from.offset = n*4;
			q->from.sym = s;
			q->reg = 4;
			q->to = p->from;
			q->to.type = D_CONST;

			q = prg();
			q->line = p->line;
			q->pc = p->pc;
			q->link = p->link;
			p->link = q;
			p = q;
			p->as = AMOVW;
			p->from.type = D_OREG;
			p->from.name = D_EXTERN;
			p->from.sym = s;
			p->from.offset = n*4 + 4;
			p->to.type = D_REG;
			p->to.reg = thumb ? REGTMPT : REGTMP;

			q = prg();
			q->line = p->line;
			q->pc = p->pc;
			q->link = p->link;
			p->link = q;
			p = q;
			p->as = AADD;
			p->from.type = D_CONST;
			p->from.offset = 1;
			p->to.type = D_REG;
			p->to.reg = thumb ? REGTMPT : REGTMP;

			q = prg();
			q->line = p->line;
			q->pc = p->pc;
			q->link = p->link;
			p->link = q;
			p = q;
			p->as = AMOVW;
			p->from.type = D_REG;
			p->from.reg = thumb ? REGTMPT : REGTMP;
			p->to.type = D_OREG;
			p->to.name = D_EXTERN;
			p->to.sym = s;
			p->to.offset = n*4 + 4;

			n += 2;
			continue;
		}
	}
	q = prg();
	q->line = 0;
	q->link = datap;
	datap = q;

	q->as = ADATA;
	q->from.type = D_OREG;
	q->from.name = D_EXTERN;
	q->from.sym = s;
	q->reg = 4;
	q->to.type = D_CONST;
	q->to.offset = n;

	s->type = SBSS;
	s->value = n*4;
}

void
doprof2(void)
{
	Sym *s2, *s4;
	Prog *p, *q, *ps2, *ps4;

	if(debug['v'])
		Bprint(&bso, "%5.2f profile 2\n", cputime());
	Bflush(&bso);
	s2 = lookup("_profin", 0);
	s4 = lookup("_profout", 0);
	if(s2->type != STEXT || s4->type != STEXT) {
		diag("_profin/_profout not defined");
		return;
	}
	ps2 = P;
	ps4 = P;
	for(p = firstp; p != P; p = p->link) {
		setarch(p);
		if(p->as == ATEXT) {
			if(p->from.sym == s2) {
				ps2 = p;
				p->reg = 1;
			}
			if(p->from.sym == s4) {
				ps4 = p;
				p->reg = 1;
			}
		}
	}
	for(p = firstp; p != P; p = p->link) {
		setarch(p);
		if(p->as == ATEXT) {
			if(p->reg & NOPROF) {
				for(;;) {
					q = p->link;
					if(q == P)
						break;
					if(q->as == ATEXT)
						break;
					p = q;
				}
				continue;
			}

			/*
			 * BL	profin, R2
			 */
			q = prg();
			q->line = p->line;
			q->pc = p->pc;
			q->link = p->link;
			p->link = q;
			p = q;
			p->as = ABL;
			p->to.type = D_BRANCH;
			p->cond = ps2;
			p->to.sym = s2;

			continue;
		}
		if(p->as == ARET) {
			/*
			 * RET
			 */
			q = prg();
			q->as = ARET;
			q->from = p->from;
			q->to = p->to;
			q->link = p->link;
			p->link = q;

			/*
			 * BL	profout
			 */
			p->as = ABL;
			p->from = zprg.from;
			p->to = zprg.to;
			p->to.type = D_BRANCH;
			p->cond = ps4;
			p->to.sym = s4;

			p = q;

			continue;
		}
	}
}

static void
puntfp(Prog *p)
{
	USED(p);
	/* floating point - punt for now */
	curtext->reg = NREG;	/* ARM */
	curtext->from.sym->thumb = 0;
	thumb = 0;
	// print("%s: generating ARM code (contains floating point ops %d)\n", curtext->from.sym->name, p->line);
}

Prog*
appendp(Prog *q)
{
	Prog *p;

	p = prg();
	p->link = q->link;
	q->link = p;
	p->line = q->line;
	return p;
}

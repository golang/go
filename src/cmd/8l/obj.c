// Inferno utils/8l/obj.c
// http://code.google.com/p/inferno-os/source/browse/utils/8l/obj.c
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
#include	"../ld/elf.h"
#include	"../ld/macho.h"
#include	"../ld/pe.h"
#include	<ar.h>

#ifndef	DEFAULT
#define	DEFAULT	'9'
#endif

char	*noname		= "<none>";
char	thechar		= '8';
char	*thestring 	= "386";

/*
 *	-H0 -T0x40004C -D0x10000000	is garbage unix
 *	-H1 -T0xd0 -R4			is unix coff
 *	-H2 -T4128 -R4096		is plan9 format
 *	-H3 -Tx -Rx			is MS-DOS .COM
 *	-H4 -Tx -Rx			is fake MS-DOS .EXE
 *	-H6 -Tx -Rx			is Apple Mach-O
 *	-H7 -Tx -Rx			is Linux ELF32
 *	-H8 -Tx -Rx			is Google Native Client
 *	-H9 -Tx -Rx			is FreeBSD ELF32
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

void
usage(void)
{
	fprint(2, "usage: 8l [-options] [-E entry] [-H head] [-L dir] [-T text] [-R rnd] [-o out] files...\n");
	exits("usage");
}

void
main(int argc, char *argv[])
{
	int i, c;

	Binit(&bso, 1, OWRITE);
	cout = -1;
	listinit();
	memset(debug, 0, sizeof(debug));
	nerrors = 0;
	outfile = nil;
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
	case 'o': /* output to (next arg) */
		outfile = EARGF(usage());
		break;
	case 'E':
		INITENTRY = EARGF(usage());
		break;
	case 'H':
		HEADTYPE = atolwhex(EARGF(usage()));
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

	mywhatsys();	// get goos

	if(HEADTYPE == -1) {
		HEADTYPE = 2;
		if(strcmp(goos, "linux") == 0)
			HEADTYPE = 7;
		else
		if(strcmp(goos, "darwin") == 0)
			HEADTYPE = 6;
		else
		if(strcmp(goos, "nacl") == 0)
			HEADTYPE = 8;
		else
		if(strcmp(goos, "freebsd") == 0)
			HEADTYPE = 9;
		else
		if(strcmp(goos, "mingw") == 0)
			HEADTYPE = 10;
		else
			print("goos is not known: %s\n", goos);
	}

	if(outfile == nil) {
		if(HEADTYPE == 10)
			outfile = "8.out.exe";
		else
			outfile = "8.out";
	}

	libinit();

	switch(HEADTYPE) {
	default:
		diag("unknown -H option");
		errorexit();

	case 0:	/* this is garbage */
		HEADR = 20L+56L;
		if(INITTEXT == -1)
			INITTEXT = 0x40004CL;
		if(INITDAT == -1)
			INITDAT = 0x10000000L;
		if(INITRND == -1)
			INITRND = 0;
		break;
	case 1:	/* is unix coff */
		HEADR = 0xd0L;
		if(INITTEXT == -1)
			INITTEXT = 0xd0;
		if(INITDAT == -1)
			INITDAT = 0x400000;
		if(INITRND == -1)
			INITRND = 0;
		break;
	case 2:	/* plan 9 */
		HEADR = 32L;
		if(INITTEXT == -1)
			INITTEXT = 4096+32;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 4096;
		break;
	case 3:	/* MS-DOS .COM */
		HEADR = 0;
		if(INITTEXT == -1)
			INITTEXT = 0x0100;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 4;
		break;
	case 4:	/* fake MS-DOS .EXE */
		HEADR = 0x200;
		if(INITTEXT == -1)
			INITTEXT = 0x0100;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 4;
		HEADR += (INITTEXT & 0xFFFF);
		if(debug['v'])
			Bprint(&bso, "HEADR = 0x%ld\n", HEADR);
		break;
	case 6:	/* apple MACH */
		/*
		 * OS X system constant - offset from %gs to our TLS.
		 * Explained in ../../libcgo/darwin_386.c.
		 */
		tlsoffset = 0x468;
		machoinit();
		HEADR = MACHORESERVE;
		if(INITTEXT == -1)
			INITTEXT = 4096+HEADR;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 4096;
		break;
	case 7:	/* elf32 executable */
	case 9:
		/*
		 * Linux ELF uses TLS offsets negative from %gs.
		 * Translate 0(GS) and 4(GS) into -8(GS) and -4(GS).
		 * Also known to ../../pkg/runtime/linux/386/sys.s
		 * and ../../libcgo/linux_386.c.
		 */
		tlsoffset = -8;
		elfinit();
		HEADR = ELFRESERVE;
		if(INITTEXT == -1)
			INITTEXT = 0x08048000+HEADR;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 4096;
		break;
	case 8:	/* native client elf32 executable */
		elfinit();
		HEADR = 4096;
		if(INITTEXT == -1)
			INITTEXT = 0x20000;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 4096;
		break;
	case 10: /* PE executable */
		peinit();
		HEADR = PERESERVE;
		if(INITTEXT == -1)
			INITTEXT = PEBASE+0x1000;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = PEALIGN;
		break;
	}
	if(INITDAT != 0 && INITRND != 0)
		print("warning: -D0x%lux is ignored because of -R0x%lux\n",
			INITDAT, INITRND);
	if(debug['v'])
		Bprint(&bso, "HEADER = -H0x%ld -T0x%lux -D0x%lux -R0x%lux\n",
			HEADTYPE, INITTEXT, INITDAT, INITRND);
	Bflush(&bso);
	for(i=1; optab[i].as; i++)
		if(i != optab[i].as) {
			diag("phase error in optab: %d", i);
			errorexit();
		}
	maxop = i;

	for(i=0; i<Ymax; i++)
		ycover[i*Ymax + i] = 1;

	ycover[Yi0*Ymax + Yi8] = 1;
	ycover[Yi1*Ymax + Yi8] = 1;

	ycover[Yi0*Ymax + Yi32] = 1;
	ycover[Yi1*Ymax + Yi32] = 1;
	ycover[Yi8*Ymax + Yi32] = 1;

	ycover[Yal*Ymax + Yrb] = 1;
	ycover[Ycl*Ymax + Yrb] = 1;
	ycover[Yax*Ymax + Yrb] = 1;
	ycover[Ycx*Ymax + Yrb] = 1;
	ycover[Yrx*Ymax + Yrb] = 1;

	ycover[Yax*Ymax + Yrx] = 1;
	ycover[Ycx*Ymax + Yrx] = 1;

	ycover[Yax*Ymax + Yrl] = 1;
	ycover[Ycx*Ymax + Yrl] = 1;
	ycover[Yrx*Ymax + Yrl] = 1;

	ycover[Yf0*Ymax + Yrf] = 1;

	ycover[Yal*Ymax + Ymb] = 1;
	ycover[Ycl*Ymax + Ymb] = 1;
	ycover[Yax*Ymax + Ymb] = 1;
	ycover[Ycx*Ymax + Ymb] = 1;
	ycover[Yrx*Ymax + Ymb] = 1;
	ycover[Yrb*Ymax + Ymb] = 1;
	ycover[Ym*Ymax + Ymb] = 1;

	ycover[Yax*Ymax + Yml] = 1;
	ycover[Ycx*Ymax + Yml] = 1;
	ycover[Yrx*Ymax + Yml] = 1;
	ycover[Yrl*Ymax + Yml] = 1;
	ycover[Ym*Ymax + Yml] = 1;

	for(i=0; i<D_NONE; i++) {
		reg[i] = -1;
		if(i >= D_AL && i <= D_BH)
			reg[i] = (i-D_AL) & 7;
		if(i >= D_AX && i <= D_DI)
			reg[i] = (i-D_AX) & 7;
		if(i >= D_F0 && i <= D_F0+7)
			reg[i] = (i-D_F0) & 7;
	}

	zprg.link = P;
	zprg.pcond = P;
	zprg.back = 2;
	zprg.as = AGOK;
	zprg.from.type = D_NONE;
	zprg.from.index = D_NONE;
	zprg.from.scale = 1;
	zprg.to = zprg.from;

	pcstr = "%.6lux ";
	nuxiinit();
	histgen = 0;
	textp = P;
	datap = P;
	edatap = P;
	pc = 0;
	dtype = 4;
	version = 0;
	cbp = buf.cbuf;
	cbc = sizeof(buf.cbuf);
	firstp = prg();
	lastp = firstp;

	while(*argv)
		objfile(*argv++);

	if(!debug['l'])
		loadlib();

	deadcode();

	firstp = firstp->link;
	if(firstp == P)
		errorexit();
	if(doexp || dlm){
		EXPTAB = "_exporttab";
		zerosig(EXPTAB);
		zerosig("etext");
		zerosig("edata");
		zerosig("end");
		if(dlm){
			import();
			HEADTYPE = 2;
			INITTEXT = INITDAT = 0;
			INITRND = 8;
			INITENTRY = EXPTAB;
		}
		export();
	}
	patch();
	follow();
	doelf();
	if(HEADTYPE == 6)
		domacho();
	dodata();
	dostkoff();
	if(debug['p'])
		if(debug['1'])
			doprof1();
		else
			doprof2();
	span();
	doinit();
	if(HEADTYPE == 10)
		dope();
	asmb();
	undef();
	if(debug['v']) {
		Bprint(&bso, "%5.2f cpu time\n", cputime());
		Bprint(&bso, "%ld symbols\n", nsymbol);
		Bprint(&bso, "%d sizeof adr\n", sizeof(Adr));
		Bprint(&bso, "%d sizeof prog\n", sizeof(Prog));
	}
	Bflush(&bso);

	errorexit();
}

void
zaddr(Biobuf *f, Adr *a, Sym *h[])
{
	int t;
	int32 l;
	Sym *s;
	Auto *u;

	t = Bgetc(f);
	a->index = D_NONE;
	a->scale = 0;
	if(t & T_INDEX) {
		a->index = Bgetc(f);
		a->scale = Bgetc(f);
	}
	a->type = D_NONE;
	a->offset = 0;
	if(t & T_OFFSET)
		a->offset = Bget4(f);
	a->offset2 = 0;
	if(t & T_OFFSET2) {
		a->offset2 = Bget4(f);
		a->type = D_CONST2;
	}
	a->sym = S;
	if(t & T_SYM)
		a->sym = h[Bgetc(f)];
	if(t & T_FCONST) {
		a->ieee.l = Bget4(f);
		a->ieee.h = Bget4(f);
		a->type = D_FCONST;
	} else
	if(t & T_SCONST) {
		Bread(f, a->scon, NSNAME);
		a->type = D_SCONST;
	}
	if(t & T_TYPE)
		a->type = Bgetc(f);
	adrgotype = S;
	if(t & T_GOTYPE)
		adrgotype = h[Bgetc(f)];

	t = a->type;
	if(t == D_INDIR+D_GS)
		a->offset += tlsoffset;

	s = a->sym;
	if(s == S)
		return;
	if(t != D_AUTO && t != D_PARAM) {
		if(adrgotype)
			s->gotype = adrgotype;
		return;
	}
	l = a->offset;
	for(u=curauto; u; u=u->link) {
		if(u->asym == s)
		if(u->type == t) {
			if(u->aoffset > l)
				u->aoffset = l;
			if(adrgotype)
				u->gotype = adrgotype;
			return;
		}
	}

	u = mal(sizeof(*u));
	u->link = curauto;
	curauto = u;
	u->asym = s;
	u->aoffset = l;
	u->type = t;
	u->gotype = adrgotype;
}

void
nopout(Prog *p)
{
	p->as = ANOP;
	p->from.type = D_NONE;
	p->to.type = D_NONE;
}

void
ldobj1(Biobuf *f, int64 len, char *pn)
{
	int32 ipc;
	Prog *p, *t;
	int v, o, r, skip;
	Sym *h[NSYM], *s, *di;
	uint32 sig;
	int ntext;
	int32 eof;
	char *name;
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
	o |= Bgetc(f) << 8;
	if(o <= AXXX || o >= ALAST) {
		if(o < 0)
			goto eof;
		diag("%s:#%lld: opcode out of range: %#ux", pn, Boffset(f), o);
		print("	probably not a .%c file\n", thechar);
		errorexit();
	}

	if(o == ANAME || o == ASIGNAME) {
		sig = 0;
		if(o == ASIGNAME)
			sig = Bget4(f);
		v = Bgetc(f);	/* type */
		o = Bgetc(f);	/* sym */
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

		if(debug['S'] && r == 0)
			sig = 1729;
		if(sig != 0){
			if(s->sig != 0 && s->sig != sig)
				diag("incompatible type signatures"
					"%lux(%s) and %lux(%s) for %s",
					s->sig, s->file, sig, pn, s->name);
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

	p = mal(sizeof(*p));
	p->as = o;
	p->line = Bget4(f);
	p->back = 2;
	p->ft = 0;
	p->tt = 0;
	zaddr(f, &p->from, h);
	fromgotype = adrgotype;
	zaddr(f, &p->to, h);

	if(debug['W'])
		print("%P\n", p);

	switch(p->as) {
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
		if(s->type == 0 || s->type == SXREF) {
			s->type = SBSS;
			s->value = 0;
		}
		if(s->type != SBSS) {
			diag("%s: redefinition: %s in %s",
				pn, s->name, TNAME);
			s->type = SBSS;
			s->value = 0;
		}
		if(p->to.offset > s->value)
			s->value = p->to.offset;
		if(p->from.scale & DUPOK)
			s->dupok = 1;
		goto loop;

	case ADYNT:
		if(p->to.sym == S) {
			diag("DYNT without a sym\n%P", p);
			break;
		}
		di = p->to.sym;
		p->from.scale = 4;
		if(di->type == SXREF) {
			if(debug['z'])
				Bprint(&bso, "%P set to %d\n", p, dtype);
			di->type = SCONST;
			di->value = dtype;
			dtype += 4;
		}
		if(p->from.sym == S)
			break;

		p->from.offset = di->value;
		p->from.sym->type = SDATA;
		if(curtext == P) {
			diag("DYNT not in text: %P", p);
			break;
		}
		p->to.sym = curtext->from.sym;
		p->to.type = D_ADDR;
		p->to.index = D_EXTERN;
		goto data;

	case AINIT:
		if(p->from.sym == S) {
			diag("INIT without a sym\n%P", p);
			break;
		}
		if(di == S) {
			diag("INIT without previous DYNT\n%P", p);
			break;
		}
		p->from.offset = di->value;
		p->from.sym->type = SDATA;
		goto data;

	case ADATA:
	data:
		// Assume that AGLOBL comes after ADATA.
		// If we've seen an AGLOBL that said this sym was DUPOK,
		// ignore any more ADATA we see, which must be
		// redefinitions.
		s = p->from.sym;
		if(s != S && s->dupok) {
//			if(debug['v'])
//				Bprint(&bso, "skipping %s in %s: dupok\n", s->name, pn);
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
		p->link = P;
		goto loop;

	case AGOK:
		diag("%s: GOK opcode in %s", pn, TNAME);
		pc++;
		goto loop;

	case ATEXT:
		s = p->from.sym;
		if(s == S) {
			diag("%s: no TEXT symbol: %P", pn, p);
			errorexit();
		}
		if(ntext++ == 0 && s->type != 0 && s->type != SXREF) {
			/* redefinition, so file has probably been seen before */
			if(debug['v'])
				diag("skipping: %s: redefinition: %s", pn, s->name);
			return;
		}
		if(curtext != P) {
			histtoauto();
			curtext->to.autom = curauto;
			curauto = 0;
		}
		skip = 0;
		curtext = p;
		if(s->type != 0 && s->type != SXREF) {
			if(p->from.scale & DUPOK) {
				skip = 1;
				goto casdef;
			}
			diag("%s: redefinition: %s\n%P", pn, s->name, p);
		}
		newtext(p, s);
		goto loop;

	case AFMOVF:
	case AFADDF:
	case AFSUBF:
	case AFSUBRF:
	case AFMULF:
	case AFDIVF:
	case AFDIVRF:
	case AFCOMF:
	case AFCOMFP:
		if(skip)
			goto casdef;
		if(p->from.type == D_FCONST) {
			/* size sb 9 max */
			sprint(literal, "$%lux", ieeedtof(&p->from.ieee));
			s = lookup(literal, 0);
			if(s->type == 0) {
				s->type = SBSS;
				s->value = 4;
				t = prg();
				t->as = ADATA;
				t->line = p->line;
				t->from.type = D_EXTERN;
				t->from.sym = s;
				t->from.scale = 4;
				t->to = p->from;
				if(edatap == P)
					datap = t;
				else
					edatap->link = t;
				edatap = t;
				t->link = P;
			}
			p->from.type = D_EXTERN;
			p->from.sym = s;
			p->from.offset = 0;
		}
		goto casdef;

	case AFMOVD:
	case AFADDD:
	case AFSUBD:
	case AFSUBRD:
	case AFMULD:
	case AFDIVD:
	case AFDIVRD:
	case AFCOMD:
	case AFCOMDP:
		if(skip)
			goto casdef;
		if(p->from.type == D_FCONST) {
			/* size sb 18 max */
			sprint(literal, "$%lux.%lux",
				p->from.ieee.l, p->from.ieee.h);
			s = lookup(literal, 0);
			if(s->type == 0) {
				s->type = SBSS;
				s->value = 8;
				t = prg();
				t->as = ADATA;
				t->line = p->line;
				t->from.type = D_EXTERN;
				t->from.sym = s;
				t->from.scale = 8;
				t->to = p->from;
				if(edatap == P)
					datap = t;
				else
					edatap->link = t;
				edatap = t;
				t->link = P;
			}
			p->from.type = D_EXTERN;
			p->from.sym = s;
			p->from.offset = 0;
		}
		goto casdef;

	casdef:
	default:
		if(skip)
			nopout(p);

		if(p->to.type == D_BRANCH)
			p->to.offset += ipc;
		lastp->link = p;
		lastp = p;
		p->pc = pc;
		pc++;
		goto loop;
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

Prog*
copyp(Prog *q)
{
	Prog *p;

	p = prg();
	*p = *q;
	return p;
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
		if(p->as == ATEXT) {
			q = prg();
			q->line = p->line;
			q->link = datap;
			datap = q;
			q->as = ADATA;
			q->from.type = D_EXTERN;
			q->from.offset = n*4;
			q->from.sym = s;
			q->from.scale = 4;
			q->to = p->from;
			q->to.type = D_CONST;

			q = prg();
			q->line = p->line;
			q->pc = p->pc;
			q->link = p->link;
			p->link = q;
			p = q;
			p->as = AADDL;
			p->from.type = D_CONST;
			p->from.offset = 1;
			p->to.type = D_EXTERN;
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
	q->from.type = D_EXTERN;
	q->from.sym = s;
	q->from.scale = 4;
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
		if(p->as == ATEXT) {
			if(p->from.sym == s2) {
				p->from.scale = 1;
				ps2 = p;
			}
			if(p->from.sym == s4) {
				p->from.scale = 1;
				ps4 = p;
			}
		}
	}
	for(p = firstp; p != P; p = p->link) {
		if(p->as == ATEXT) {
			curtext = p;

			if(p->from.scale & NOPROF) {	/* dont profile */
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
			 * JMPL	profin
			 */
			q = prg();
			q->line = p->line;
			q->pc = p->pc;
			q->link = p->link;
			p->link = q;
			p = q;
			p->as = ACALL;
			p->to.type = D_BRANCH;
			p->pcond = ps2;
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
			 * JAL	profout
			 */
			p->as = ACALL;
			p->from = zprg.from;
			p->to = zprg.to;
			p->to.type = D_BRANCH;
			p->pcond = ps4;
			p->to.sym = s4;

			p = q;

			continue;
		}
	}
}

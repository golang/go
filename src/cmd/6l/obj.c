// Inferno utils/6l/obj.c
// http://code.google.com/p/inferno-os/source/browse/utils/6l/obj.c
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
#include	<ar.h>

char	*noname		= "<none>";
char	thechar		= '6';
char*	thestring 	= "amd64";
char*	paramspace	= "FP";

/*
 *	-H2 -T4136 -R4096		is plan9 64-bit format
 *	-H3 -T4128 -R4096		is plan9 32-bit format
 *	-H5 -T0x80110000 -R4096		is ELF32
 *	-H6 -Tx -Rx			is apple MH-exec
 *	-H7 -Tx -Rx			is linux elf-exec
 *      -H9 -Tx -Rx			is FreeBSD elf-exec
 *
 *	options used: 189BLQSWabcjlnpsvz
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
	else {
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
	fprint(2, "usage: 6l [-options] [-E entry] [-H head] [-L dir] [-T text] [-R rnd] [-o out] main.6\n");
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
	outfile = "6.out";
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

	if(argc != 1)
		usage();

	libinit();

	if(HEADTYPE == -1) {
		HEADTYPE = 2;
		if(strcmp(goos, "linux") == 0)
			HEADTYPE = 7;
		else
		if(strcmp(goos, "darwin") == 0)
			HEADTYPE = 6;
		else
		if(strcmp(goos, "freebsd") == 0)
			HEADTYPE = 9;
		else
			print("goos is not known: %s\n", goos);
	}

	switch(HEADTYPE) {
	default:
		diag("unknown -H option");
		errorexit();
	case 2:	/* plan 9 */
		HEADR = 32L+8L;
		if(INITTEXT == -1)
			INITTEXT = 4096+HEADR;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 4096;
		break;
	case 3:	/* plan 9 */
		HEADR = 32L;
		if(INITTEXT == -1)
			INITTEXT = 4096+32;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 4096;
		break;
	case 5:	/* elf32 executable */
		HEADR = rnd(52L+3*32L, 16);
		if(INITTEXT == -1)
			INITTEXT = 0x80110000L;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 4096;
		break;
	case 6:	/* apple MACH */
		machoinit();
		HEADR = MACHORESERVE;
		if(INITRND == -1)
			INITRND = 4096;
		if(INITTEXT == -1)
			INITTEXT = 4096+HEADR;
		if(INITDAT == -1)
			INITDAT = 0;
		break;
	case 7:	/* elf64 executable */
	case 9: /* freebsd */
		elfinit();
		HEADR = ELFRESERVE;
		if(INITTEXT == -1)
			INITTEXT = (1<<22)+HEADR;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 4096;
		break;
	}
	if(INITDAT != 0 && INITRND != 0)
		print("warning: -D0x%llux is ignored because of -R0x%lux\n",
			INITDAT, INITRND);
	if(debug['v'])
		Bprint(&bso, "HEADER = -H%ld -T0x%llux -D0x%llux -R0x%lux\n",
			HEADTYPE, INITTEXT, INITDAT, INITRND);
	Bflush(&bso);
	for(i=1; optab[i].as; i++) {
		c = optab[i].as;
		if(opindex[c] != nil) {
			diag("phase error in optab: %d (%A)", i, c);
			errorexit();
		}
		opindex[c] = &optab[i];
	}

	for(i=0; i<Ymax; i++)
		ycover[i*Ymax + i] = 1;

	ycover[Yi0*Ymax + Yi8] = 1;
	ycover[Yi1*Ymax + Yi8] = 1;

	ycover[Yi0*Ymax + Ys32] = 1;
	ycover[Yi1*Ymax + Ys32] = 1;
	ycover[Yi8*Ymax + Ys32] = 1;

	ycover[Yi0*Ymax + Yi32] = 1;
	ycover[Yi1*Ymax + Yi32] = 1;
	ycover[Yi8*Ymax + Yi32] = 1;
	ycover[Ys32*Ymax + Yi32] = 1;

	ycover[Yi0*Ymax + Yi64] = 1;
	ycover[Yi1*Ymax + Yi64] = 1;
	ycover[Yi8*Ymax + Yi64] = 1;
	ycover[Ys32*Ymax + Yi64] = 1;
	ycover[Yi32*Ymax + Yi64] = 1;

	ycover[Yal*Ymax + Yrb] = 1;
	ycover[Ycl*Ymax + Yrb] = 1;
	ycover[Yax*Ymax + Yrb] = 1;
	ycover[Ycx*Ymax + Yrb] = 1;
	ycover[Yrx*Ymax + Yrb] = 1;
	ycover[Yrl*Ymax + Yrb] = 1;

	ycover[Ycl*Ymax + Ycx] = 1;

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
	ycover[Yrl*Ymax + Ymb] = 1;
	ycover[Ym*Ymax + Ymb] = 1;

	ycover[Yax*Ymax + Yml] = 1;
	ycover[Ycx*Ymax + Yml] = 1;
	ycover[Yrx*Ymax + Yml] = 1;
	ycover[Yrl*Ymax + Yml] = 1;
	ycover[Ym*Ymax + Yml] = 1;

	ycover[Yax*Ymax + Ymm] = 1;
	ycover[Ycx*Ymax + Ymm] = 1;
	ycover[Yrx*Ymax + Ymm] = 1;
	ycover[Yrl*Ymax + Ymm] = 1;
	ycover[Ym*Ymax + Ymm] = 1;
	ycover[Ymr*Ymax + Ymm] = 1;

	ycover[Yax*Ymax + Yxm] = 1;
	ycover[Ycx*Ymax + Yxm] = 1;
	ycover[Yrx*Ymax + Yxm] = 1;
	ycover[Yrl*Ymax + Yxm] = 1;
	ycover[Ym*Ymax + Yxm] = 1;
	ycover[Yxr*Ymax + Yxm] = 1;

	for(i=0; i<D_NONE; i++) {
		reg[i] = -1;
		if(i >= D_AL && i <= D_R15B) {
			reg[i] = (i-D_AL) & 7;
			if(i >= D_SPB && i <= D_DIB)
				regrex[i] = 0x40;
			if(i >= D_R8B && i <= D_R15B)
				regrex[i] = Rxr | Rxx | Rxb;
		}
		if(i >= D_AH && i<= D_BH)
			reg[i] = 4 + ((i-D_AH) & 7);
		if(i >= D_AX && i <= D_R15) {
			reg[i] = (i-D_AX) & 7;
			if(i >= D_R8)
				regrex[i] = Rxr | Rxx | Rxb;
		}
		if(i >= D_F0 && i <= D_F0+7)
			reg[i] = (i-D_F0) & 7;
		if(i >= D_M0 && i <= D_M0+7)
			reg[i] = (i-D_M0) & 7;
		if(i >= D_X0 && i <= D_X0+15) {
			reg[i] = (i-D_X0) & 7;
			if(i >= D_X0+8)
				regrex[i] = Rxr | Rxx | Rxb;
		}
		if(i >= D_CR+8 && i <= D_CR+15)
			regrex[i] = Rxr;
	}

	zprg.link = P;
	zprg.pcond = P;
	zprg.back = 2;
	zprg.as = AGOK;
	zprg.from.type = D_NONE;
	zprg.from.index = D_NONE;
	zprg.from.scale = 1;
	zprg.to = zprg.from;
	zprg.mode = 64;

	pcstr = "%.6llux ";
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

	objfile(argv[0], "main");

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
			INITTEXT = 0;
			INITDAT = 0;
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
	dobss();
	dostkoff();
	paramspace = "SP";	/* (FP) now (SP) on output */
	if(debug['p'])
		if(debug['1'])
			doprof1();
		else
			doprof2();
	span();
	doinit();
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
	a->offset = 0;
	if(t & T_OFFSET) {
		a->offset = Bget4(f);
		if(t & T_64) {
			a->offset &= 0xFFFFFFFFULL;
			a->offset |= (vlong)Bget4(f) << 32;
		}
	}
	a->sym = S;
	if(t & T_SYM)
		a->sym = h[Bgetc(f)];
	a->type = D_NONE;
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
	s = a->sym;
	if(s == S)
		return;

	t = a->type;
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
ldobj1(Biobuf *f, char *pkg, int64 len, char *pn)
{
	vlong ipc;
	Prog *p, *t;
	int v, o, r, skip, mode;
	Sym *h[NSYM], *s, *di;
	uint32 sig;
	char *name, *x;
	int ntext;
	vlong eof;
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
	mode = 64;

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
		print("	probably not a .6 file\n");
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
		x = expandpkg(name, pkg);
		s = lookup(x, r);
		if(x != name)
			free(x);
		name = nil;

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
	p->mode = mode;
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
			if(s->file == nil)
				s->file = pn;
			else if(s->file != pn) {
				diag("multiple initialization for %s: in both %s and %s", s->name, s->file, pn);
				errorexit();
			}
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
		if(ntext++ == 0 && s->type != 0 && s->type != SXREF) {
			/* redefinition, so file has probably been seen before */
			if(debug['v'])
				Bprint(&bso, "skipping: %s: redefinition: %s", pn, s->name);
			return;
		}
		if(curtext != P) {
			histtoauto();
			curtext->to.autom = curauto;
			curauto = 0;
		}
		skip = 0;
		curtext = p;
		if(s == S) {
			diag("%s: no TEXT symbol: %P", pn, p);
			errorexit();
		}
		if(s->type != 0 && s->type != SXREF) {
			if(p->from.scale & DUPOK) {
				skip = 1;
				goto casdef;
			}
			diag("%s: redefinition: %s\n%P", pn, s->name, p);
		}
		if(fromgotype) {
			if(s->gotype && s->gotype != fromgotype)
				diag("%s: type mismatch for %s", pn, s->name);
			s->gotype = fromgotype;
		}
		newtext(p, s);
		goto loop;

	case AMODE:
		if(p->from.type == D_CONST || p->from.type == D_INDIR+D_NONE){
			switch((int)p->from.offset){
			case 16: case 32: case 64:
				mode = p->from.offset;
				break;
			}
		}
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
	case AMOVSS:
	case AADDSS:
	case ASUBSS:
	case AMULSS:
	case ADIVSS:
	case ACOMISS:
	case AUCOMISS:
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
	case AMOVSD:
	case AADDSD:
	case ASUBSD:
	case AMULSD:
	case ADIVSD:
	case ACOMISD:
	case AUCOMISD:
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

	p = mal(sizeof(*p));

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
	p->mode = q->mode;
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


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

// Reading object files.

#define	EXTERN
#include	"l.h"
#include	"../ld/lib.h"
#include	"../ld/elf.h"
#include	"../ld/macho.h"
#include	"../ld/dwarf.h"
#include	"../ld/pe.h"
#include	<ar.h>

#ifndef	DEFAULT
#define	DEFAULT	'9'
#endif

char	*noname		= "<none>";
char	*thestring 	= "386";

Header headers[] = {
	"garbunix", Hgarbunix,
	"unixcoff", Hunixcoff,
	"plan9", Hplan9x32,
	"msdoscom", Hmsdoscom,
	"msdosexe", Hmsdosexe,
	"darwin", Hdarwin,
	"linux", Hlinux,
	"freebsd", Hfreebsd,
	"netbsd", Hnetbsd,
	"openbsd", Hopenbsd,
	"windows", Hwindows,
	"windowsgui", Hwindows,
	0, 0
};

/*
 *	-Hgarbunix -T0x40004C -D0x10000000	is garbage unix
 *	-Hunixcoff -T0xd0 -R4			is unix coff
 *	-Hplan9 -T4128 -R4096			is plan9 format
 *	-Hmsdoscom -Tx -Rx			is MS-DOS .COM
 *	-Hmsdosexe -Tx -Rx			is fake MS-DOS .EXE
 *	-Hdarwin -Tx -Rx			is Apple Mach-O
 *	-Hlinux -Tx -Rx				is Linux ELF32
 *	-Hfreebsd -Tx -Rx			is FreeBSD ELF32
 *	-Hnetbsd -Tx -Rx			is NetBSD ELF32
 *	-Hopenbsd -Tx -Rx			is OpenBSD ELF32
 *	-Hwindows -Tx -Rx			is MS Windows PE32
 */

void
main(int argc, char *argv[])
{
	Binit(&bso, 1, OWRITE);
	listinit();
	memset(debug, 0, sizeof(debug));
	nerrors = 0;
	outfile = nil;
	HEADTYPE = -1;
	INITTEXT = -1;
	INITDAT = -1;
	INITRND = -1;
	INITENTRY = 0;
	LIBINITENTRY = 0;
	linkmode = LinkAuto;
	nuxiinit();

	flagcount("1", "use alternate profiling code", &debug['1']);
	flagfn1("B", "info: define ELF NT_GNU_BUILD_ID note", addbuildinfo);
	flagstr("E", "sym: entry symbol", &INITENTRY);
	flagint32("D", "addr: data address", &INITDAT);
	flagfn1("I", "interp: set ELF interp", setinterp);
	flagfn1("L", "dir: add dir to library path", Lflag);
	flagfn1("H", "head: header type", setheadtype);
	flagcount("K", "add stack underflow checks", &debug['K']);
	flagcount("O", "print pc-line tables", &debug['O']);
	flagcount("Q", "debug byte-register code gen", &debug['Q']);
	flagint32("R", "rnd: address rounding", &INITRND);
	flagcount("S", "check type signatures", &debug['S']);
	flagint32("T", "addr: text address", &INITTEXT);
	flagfn0("V", "print version and exit", doversion);
	flagcount("W", "disassemble input", &debug['W']);
	flagfn2("X", "name value: define string data", addstrdata);
	flagcount("Z", "clear stack frame on entry", &debug['Z']);
	flagcount("a", "disassemble output", &debug['a']);
	flagcount("c", "dump call graph", &debug['c']);
	flagcount("d", "disable dynamic executable", &debug['d']);
	flagstr("extld", "linker to run in external mode", &extld);
	flagstr("extldflags", "flags for external linker", &extldflags);
	flagcount("f", "ignore version mismatch", &debug['f']);
	flagcount("g", "disable go package data checks", &debug['g']);
	flagfn1("linkmode", "mode: set link mode (internal, external, auto)", setlinkmode);
	flagstr("k", "sym: set field tracking symbol", &tracksym);
	flagstr("o", "outfile: set output file", &outfile);
	flagcount("p", "insert profiling code", &debug['p']);
	flagstr("r", "dir1:dir2:...: set ELF dynamic linker search path", &rpath);
	flagcount("race", "enable race detector", &flag_race);
	flagcount("s", "disable symbol table", &debug['s']);
	flagcount("n", "dump symbol table", &debug['n']);
	flagstr("tmpdir", "leave temporary files in this directory", &tmpdir);
	flagcount("u", "reject unsafe packages", &debug['u']);
	flagcount("v", "print link trace", &debug['v']);
	flagcount("w", "disable DWARF generation", &debug['w']);
	// TODO: link mode flag
	
	flagparse(&argc, &argv, usage);

	if(argc != 1)
		usage();

	mywhatsys();	// get goos

	if(HEADTYPE == -1)
		HEADTYPE = headtype(goos);

	// getgoextlinkenabled is based on GO_EXTLINK_ENABLED when
	// Go was built; see ../../make.bash.
	if(linkmode == LinkAuto && strcmp(getgoextlinkenabled(), "0") == 0)
		linkmode = LinkInternal;

	switch(HEADTYPE) {
	default:
		if(linkmode == LinkAuto)
			linkmode = LinkInternal;
		if(linkmode == LinkExternal && strcmp(getgoextlinkenabled(), "1") != 0)
			sysfatal("cannot use -linkmode=external with -H %s", headstr(HEADTYPE));
		break;
	case Hdarwin:
	case Hfreebsd:
	case Hlinux:
	case Hnetbsd:
	case Hopenbsd:
		break;
	}

	if(outfile == nil) {
		if(HEADTYPE == Hwindows)
			outfile = "8.out.exe";
		else
			outfile = "8.out";
	}

	libinit();

	switch(HEADTYPE) {
	default:
		diag("unknown -H option");
		errorexit();

	case Hgarbunix:	/* this is garbage */
		HEADR = 20L+56L;
		if(INITTEXT == -1)
			INITTEXT = 0x40004CL;
		if(INITDAT == -1)
			INITDAT = 0x10000000L;
		if(INITRND == -1)
			INITRND = 0;
		break;
	case Hunixcoff:	/* is unix coff */
		HEADR = 0xd0L;
		if(INITTEXT == -1)
			INITTEXT = 0xd0;
		if(INITDAT == -1)
			INITDAT = 0x400000;
		if(INITRND == -1)
			INITRND = 0;
		break;
	case Hplan9x32:	/* plan 9 */
		tlsoffset = -8;
		HEADR = 32L;
		if(INITTEXT == -1)
			INITTEXT = 4096+32;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 4096;
		break;
	case Hmsdoscom:	/* MS-DOS .COM */
		HEADR = 0;
		if(INITTEXT == -1)
			INITTEXT = 0x0100;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 4;
		break;
	case Hmsdosexe:	/* fake MS-DOS .EXE */
		HEADR = 0x200;
		if(INITTEXT == -1)
			INITTEXT = 0x0100;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 4;
		HEADR += (INITTEXT & 0xFFFF);
		if(debug['v'])
			Bprint(&bso, "HEADR = 0x%d\n", HEADR);
		break;
	case Hdarwin:	/* apple MACH */
		/*
		 * OS X system constant - offset from %gs to our TLS.
		 * Explained in ../../pkg/runtime/cgo/gcc_darwin_386.c.
		 */
		tlsoffset = 0x468;
		machoinit();
		HEADR = INITIAL_MACHO_HEADR;
		if(INITTEXT == -1)
			INITTEXT = 4096+HEADR;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 4096;
		break;
	case Hlinux:	/* elf32 executable */
	case Hfreebsd:
	case Hnetbsd:
	case Hopenbsd:
		/*
		 * ELF uses TLS offsets negative from %gs.
		 * Translate 0(GS) and 4(GS) into -8(GS) and -4(GS).
		 * Also known to ../../pkg/runtime/sys_linux_386.s
		 * and ../../pkg/runtime/cgo/gcc_linux_386.c.
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
	case Hwindows: /* PE executable */
		peinit();
		HEADR = PEFILEHEADR;
		if(INITTEXT == -1)
			INITTEXT = PEBASE+PESECTHEADR;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = PESECTALIGN;
		break;
	}
	if(INITDAT != 0 && INITRND != 0)
		print("warning: -D0x%ux is ignored because of -R0x%ux\n",
			INITDAT, INITRND);
	if(debug['v'])
		Bprint(&bso, "HEADER = -H0x%d -T0x%ux -D0x%ux -R0x%ux\n",
			HEADTYPE, INITTEXT, INITDAT, INITRND);
	Bflush(&bso);

	instinit();
	zprg.link = P;
	zprg.pcond = P;
	zprg.back = 2;
	zprg.as = AGOK;
	zprg.from.type = D_NONE;
	zprg.from.index = D_NONE;
	zprg.from.scale = 1;
	zprg.to = zprg.from;

	pcstr = "%.6ux ";
	histgen = 0;
	pc = 0;
	dtype = 4;
	version = 0;
	cbp = buf.cbuf;
	cbc = sizeof(buf.cbuf);

	addlibpath("command line", "command line", argv[0], "main");
	loadlib();
	deadcode();
	patch();
	follow();
	doelf();
	if(HEADTYPE == Hdarwin)
		domacho();
	if(HEADTYPE == Hwindows)
		dope();
	dostkoff();
	dostkcheck();
	if(debug['p'])
		if(debug['1'])
			doprof1();
		else
			doprof2();
	span();
	addexport();
	textaddress();
	pclntab();
	symtab();
	dodata();
	address();
	doweak();
	reloc();
	asmb();
	undef();
	hostlink();

	if(debug['v']) {
		Bprint(&bso, "%5.2f cpu time\n", cputime());
		Bprint(&bso, "%d symbols\n", nsymbol);
		Bprint(&bso, "%d sizeof adr\n", sizeof(Adr));
		Bprint(&bso, "%d sizeof prog\n", sizeof(Prog));
	}
	Bflush(&bso);

	errorexit();
}

static Sym*
zsym(char *pn, Biobuf *f, Sym *h[])
{	
	int o;
	
	o = BGETC(f);
	if(o < 0 || o >= NSYM || h[o] == nil)
		mangle(pn);
	return h[o];
}

static void
zaddr(char *pn, Biobuf *f, Adr *a, Sym *h[])
{
	int t;
	int32 l;
	Sym *s;
	Auto *u;

	t = BGETC(f);
	a->index = D_NONE;
	a->scale = 0;
	if(t & T_INDEX) {
		a->index = BGETC(f);
		a->scale = BGETC(f);
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
		a->sym = zsym(pn, f, h);
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
		a->type = BGETC(f);
	adrgotype = S;
	if(t & T_GOTYPE)
		adrgotype = zsym(pn, f, h);

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
ldobj1(Biobuf *f, char *pkg, int64 len, char *pn)
{
	int32 ipc;
	Prog *p;
	int v, o, r, skip;
	Sym *h[NSYM], *s;
	uint32 sig;
	int ntext;
	int32 eof;
	char *name, *x;
	char src[1024];
	Prog *lastp;

	lastp = nil;
	ntext = 0;
	eof = Boffset(f) + len;
	src[0] = 0;
	pn = estrdup(pn); // we keep it in Sym* references

newloop:
	memset(h, 0, sizeof(h));
	version++;
	histfrogp = 0;
	ipc = pc;
	skip = 0;

loop:
	if(f->state == Bracteof || Boffset(f) >= eof)
		goto eof;
	o = BGETC(f);
	if(o == Beof)
		goto eof;
	o |= BGETC(f) << 8;
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
		v = BGETC(f);	/* type */
		o = BGETC(f);	/* sym */
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

		if(debug['S'] && r == 0)
			sig = 1729;
		if(sig != 0){
			if(s->sig != 0 && s->sig != sig)
				diag("incompatible type signatures "
					"%ux(%s) and %ux(%s) for %s",
					s->sig, s->file, sig, pn, s->name);
			s->sig = sig;
			s->file = pn;
		}

		if(debug['W'])
			print("	ANAME	%s\n", s->name);
		if(o < 0 || o >= nelem(h))
			mangle(pn);
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
			dwarfaddfrag(s->value, s->name);
		}
		goto loop;
	}

	p = mal(sizeof(*p));
	p->as = o;
	p->line = Bget4(f);
	p->back = 2;
	zaddr(pn, f, &p->from, h);
	fromgotype = adrgotype;
	zaddr(pn, f, &p->to, h);

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
		if(cursym != nil && cursym->text)
			cursym->autom = curauto;
		curauto = 0;
		cursym = nil;
		if(Boffset(f) == eof)
			return;
		goto newloop;

	case AGLOBL:
		s = p->from.sym;
		if(s->type == 0 || s->type == SXREF) {
			s->type = SBSS;
			s->size = 0;
		}
		if(s->type != SBSS && s->type != SNOPTRBSS && !s->dupok) {
			diag("%s: redefinition: %s in %s",
				pn, s->name, TNAME);
			s->type = SBSS;
			s->size = 0;
		}
		if(p->to.offset > s->size)
			s->size = p->to.offset;
		if(p->from.scale & DUPOK)
			s->dupok = 1;
		if(p->from.scale & RODATA)
			s->type = SRODATA;
		else if(p->from.scale & NOPTR)
			s->type = SNOPTRBSS;
		goto loop;

	case ADATA:
		// Assume that AGLOBL comes after ADATA.
		// If we've seen an AGLOBL that said this sym was DUPOK,
		// ignore any more ADATA we see, which must be
		// redefinitions.
		s = p->from.sym;
		if(s->dupok) {
//			if(debug['v'])
//				Bprint(&bso, "skipping %s in %s: dupok\n", s->name, pn);
			goto loop;
		}
		if(s->file == nil)
			s->file = pn;
		else if(s->file != pn) {
			diag("multiple initialization for %s: in both %s and %s", s->name, s->file, pn);
			errorexit();
		}
		savedata(s, p, pn);
		unmal(p, sizeof *p);
		goto loop;

	case AGOK:
		diag("%s: GOK opcode in %s", pn, TNAME);
		pc++;
		goto loop;

	case ALOCALS:
		if(skip)
			goto casdef;
		cursym->locals = p->to.offset;
		pc++;
		goto loop;

	case ATYPE:
		if(skip)
			goto casdef;
		pc++;
		goto loop;

	case ATEXT:
		s = p->from.sym;
		if(s->text != nil) {
			if(p->from.scale & DUPOK) {
				skip = 1;
				goto casdef;
			}
			diag("%s: %s: redefinition", pn, s->name);
			return;
		}
		if(ntext++ == 0 && s->type != 0 && s->type != SXREF) {
			/* redefinition, so file has probably been seen before */
			if(debug['v'])
				diag("skipping: %s: redefinition: %s", pn, s->name);
			return;
		}
		if(cursym != nil && cursym->text) {
			histtoauto();
			cursym->autom = curauto;
			curauto = 0;
		}
		skip = 0;
		if(etextp)
			etextp->next = s;
		else
			textp = s;
		etextp = s;
		s->text = p;
		cursym = s;
		if(s->type != 0 && s->type != SXREF) {
			if(p->from.scale & DUPOK) {
				skip = 1;
				goto casdef;
			}
			diag("%s: redefinition: %s\n%P", pn, s->name, p);
		}
		s->type = STEXT;
		s->value = pc;
		s->args = p->to.offset2;
		lastp = p;
		p->pc = pc++;
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
			sprint(literal, "$%ux", ieeedtof(&p->from.ieee));
			s = lookup(literal, 0);
			if(s->type == 0) {
				s->type = SRODATA;
				adduint32(s, ieeedtof(&p->from.ieee));
				s->reachable = 0;
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
			sprint(literal, "$%ux.%ux",
				p->from.ieee.l, p->from.ieee.h);
			s = lookup(literal, 0);
			if(s->type == 0) {
				s->type = SRODATA;
				adduint32(s, p->from.ieee.l);
				adduint32(s, p->from.ieee.h);
				s->reachable = 0;
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
		p->pc = pc;
		pc++;

		if(p->to.type == D_BRANCH)
			p->to.offset += ipc;
		if(lastp == nil) {
			if(p->as != ANOP)
				diag("unexpected instruction: %P", p);
			goto loop;
		}
		lastp->link = p;
		lastp = p;
		goto loop;
	}

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

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

// Reading object files.

#define	EXTERN
#include	"l.h"
#include	"../ld/lib.h"
#include	"../ld/elf.h"
#include	"../ld/dwarf.h"
#include	<ar.h>

#ifndef	DEFAULT
#define	DEFAULT	'9'
#endif

char	*noname		= "<none>";
char	*thestring 	= "arm";

Header headers[] = {
   "noheader", Hnoheader,
   "risc", Hrisc,
   "plan9", Hplan9x32,
   "ixp1200", Hixp1200,
   "ipaq", Hipaq,
   "linux", Hlinux,
   "freebsd", Hfreebsd,
   "netbsd", Hnetbsd,
   0, 0
};

/*
 *	-Hrisc -T0x10005000 -R4		is aif for risc os
 *	-Hplan9 -T4128 -R4096		is plan9 format
 *	-Hixp1200			is IXP1200 (raw)
 *	-Hipaq -T0xC0008010 -R1024	is ipaq
 *	-Hlinux -Tx -Rx			is linux elf
 *	-Hfreebsd			is freebsd elf
 *	-Hnetbsd			is netbsd elf
 */

void
main(int argc, char *argv[])
{
	char *p;
	Sym *s;

	Binit(&bso, 1, OWRITE);
	listinit();
	nerrors = 0;
	outfile = "5.out";
	HEADTYPE = -1;
	INITTEXT = -1;
	INITDAT = -1;
	INITRND = -1;
	INITENTRY = 0;
	LIBINITENTRY = 0;
	linkmode = LinkInternal; // TODO: LinkAuto once everything works.
	nuxiinit();
	
	p = getgoarm();
	if(p != nil)
		goarm = atoi(p);
	else
		goarm = 6;
	if(goarm == 5)
		debug['F'] = 1;

	flagcount("1", "use alternate profiling code", &debug['1']);
	flagfn1("B", "info: define ELF NT_GNU_BUILD_ID note", addbuildinfo);
	flagstr("E", "sym: entry symbol", &INITENTRY);
	flagint32("D", "addr: data address", &INITDAT);
	flagcount("G", "debug pseudo-ops", &debug['G']);
	flagfn1("I", "interp: set ELF interp", setinterp);
	flagfn1("L", "dir: add dir to library path", Lflag);
	flagfn1("H", "head: header type", setheadtype);
	flagcount("K", "add stack underflow checks", &debug['K']);
	flagcount("M", "disable software div/mod", &debug['M']);
	flagcount("O", "print pc-line tables", &debug['O']);
	flagcount("P", "debug code generation", &debug['P']);
	flagint32("R", "rnd: address rounding", &INITRND);
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
	flagstr("k", "sym: set field tracking symbol", &tracksym);
	flagfn1("linkmode", "mode: set link mode (internal, external, auto)", setlinkmode);
	flagcount("n", "dump symbol table", &debug['n']);
	flagstr("o", "outfile: set output file", &outfile);
	flagcount("p", "insert profiling code", &debug['p']);
	flagstr("r", "dir1:dir2:...: set ELF dynamic linker search path", &rpath);
	flagcount("race", "enable race detector", &flag_race);
	flagcount("s", "disable symbol table", &debug['s']);
	flagstr("tmpdir", "leave temporary files in this directory", &tmpdir);
	flagcount("u", "reject unsafe packages", &debug['u']);
	flagcount("v", "print link trace", &debug['v']);
	flagcount("w", "disable DWARF generation", &debug['w']);
	flagcount("shared", "generate shared object", &flag_shared);
	// TODO: link mode flag
	
	flagparse(&argc, &argv, usage);

	if(argc != 1)
		usage();

	// getgoextlinkenabled is based on GO_EXTLINK_ENABLED when
	// Go was built; see ../../make.bash.
	if(linkmode == LinkAuto && strcmp(getgoextlinkenabled(), "0") == 0)
		linkmode = LinkInternal;

	if(linkmode == LinkExternal) {
		diag("only -linkmode=internal is supported");
		errorexit();
	} else if(linkmode == LinkAuto) {
		linkmode = LinkInternal;
	}

	libinit();

	if(HEADTYPE == -1)
		HEADTYPE = headtype(goos);
	switch(HEADTYPE) {
	default:
		diag("unknown -H option");
		errorexit();
	case Hnoheader:	/* no header */
		HEADR = 0L;
		if(INITTEXT == -1)
			INITTEXT = 0;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 4;
		break;
	case Hrisc:	/* aif for risc os */
		HEADR = 128L;
		if(INITTEXT == -1)
			INITTEXT = 0x10005000 + HEADR;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 4;
		break;
	case Hplan9x32:	/* plan 9 */
		HEADR = 32L;
		if(INITTEXT == -1)
			INITTEXT = 4128;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 4096;
		break;
	case Hixp1200: /* boot for IXP1200 */
		HEADR = 0L;
		if(INITTEXT == -1)
			INITTEXT = 0x0;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 4;
		break;
	case Hipaq: /* boot for ipaq */
		HEADR = 16L;
		if(INITTEXT == -1)
			INITTEXT = 0xC0008010;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 1024;
		break;
	case Hlinux:	/* arm elf */
	case Hfreebsd:
	case Hnetbsd:
		debug['d'] = 0;	// with dynamic linking
		tlsoffset = -8; // hardcoded number, first 4-byte word for g, and then 4-byte word for m
		                // this number is known to ../../pkg/runtime/cgo/gcc_linux_arm.c
		elfinit();
		HEADR = ELFRESERVE;
		if(INITTEXT == -1)
			INITTEXT = 0x10000 + HEADR;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 4096;
		break;
	}
	if(INITDAT != 0 && INITRND != 0)
		print("warning: -D0x%ux is ignored because of -R0x%ux\n",
			INITDAT, INITRND);
	if(debug['v'])
		Bprint(&bso, "HEADER = -H0x%d -T0x%ux -D0x%ux -R0x%ux\n",
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
	histgen = 0;
	pc = 0;
	dtype = 4;

	version = 0;
	cbp = buf.cbuf;
	cbc = sizeof(buf.cbuf);

	// embed goarm to runtime.goarm
	s = lookup("runtime.goarm", 0);
	s->dupok = 1;
	adduint8(s, goarm);

	addlibpath("command line", "command line", argv[0], "main");
	loadlib();

	// mark some functions that are only referenced after linker code editing
	if(debug['F'])
		mark(rlookup("_sfloat", 0));
	deadcode();
	if(textp == nil) {
		diag("no code");
		errorexit();
	}

	patch();
	if(debug['p'])
		if(debug['1'])
			doprof1();
		else
			doprof2();
	doelf();
	follow();
	softfloat();
	// 5l -Z means zero the stack frame on entry.
	// This slows down function calls but can help avoid
	// false positives in garbage collection.
	if(debug['Z'])
		dozerostk();
	noops(); // generate stack split prolog, handle div/mod, etc.
	dostkcheck();
	span();
	addexport();
	// textaddress() functionality is handled in span()
	pclntab();
	symtab();
	dodata();
	address();
	doweak();
	reloc();
	asmb();
	undef();
	hostlink();

	if(debug['c'])
		print("ARM size = %d\n", armsize);
	if(debug['v']) {
		Bprint(&bso, "%5.2f cpu time\n", cputime());
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
	if(o == 0)
		return S;
	if(o < 0 || o >= NSYM || h[o] == nil)
		mangle(pn);
	return h[o];
}

static void
zaddr(char *pn, Biobuf *f, Adr *a, Sym *h[])
{
	int i, c;
	int32 l;
	Sym *s;
	Auto *u;

	a->type = BGETC(f);
	a->reg = BGETC(f);
	c = BGETC(f);
	if(c < 0 || c > NSYM){
		print("sym out of range: %d\n", c);
		Bputc(f, ALAST+1);
		return;
	}
	a->sym = h[c];
	a->name = BGETC(f);
	adrgotype = zsym(pn, f, h);

	if((schar)a->reg < 0 || a->reg > NREG) {
		print("register out of range %d\n", a->reg);
		Bputc(f, ALAST+1);
		return;	/*  force real diagnostic */
	}

	if(a->type == D_CONST || a->type == D_OCONST) {
		if(a->name == D_EXTERN || a->name == D_STATIC) {
			s = a->sym;
			if(s != S && (s->type == STEXT || s->type == SCONST || s->type == SXREF)) {
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
	case D_REGREG2:
		a->offset = BGETC(f);
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
		break;

	case D_FCONST:
		a->ieee.l = Bget4(f);
		a->ieee.h = Bget4(f);
		break;
	}
	s = a->sym;
	if(s == S)
		return;
	i = a->name;
	if(i != D_AUTO && i != D_PARAM) {
		if(s && adrgotype)
			s->gotype = adrgotype;
		return;
	}

	l = a->offset;
	for(u=curauto; u; u=u->link)
		if(u->asym == s)
		if(u->type == i) {
			if(u->aoffset > l)
				u->aoffset = l;
			if(adrgotype)
				u->gotype = adrgotype;
			return;
		}

	u = mal(sizeof(Auto));
	u->link = curauto;
	curauto = u;
	u->asym = s;
	u->aoffset = l;
	u->type = i;
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
	Sym *h[NSYM], *s;
	int v, o, r, skip;
	uint32 sig;
	char *name;
	int ntext;
	int32 eof;
	char src[1024], *x;
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

	if(o <= AXXX || o >= ALAST) {
		diag("%s:#%lld: opcode out of range: %#ux", pn, Boffset(f), o);
		print("	probably not a .5 file\n");
		errorexit();
	}
	if(o == ANAME || o == ASIGNAME) {
		sig = 0;
		if(o == ASIGNAME)
			sig = Bget4(f);
		v = BGETC(f); /* type */
		o = BGETC(f); /* sym */
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

		if(sig != 0){
			if(s->sig != 0 && s->sig != sig)
				diag("incompatible type signatures %ux(%s) and %ux(%s) for %s", s->sig, s->file, sig, pn, s->name);
			s->sig = sig;
			s->file = pn;
		}

		if(debug['W'])
			print("	ANAME	%s\n", s->name);
		if(o < 0 || o >= nelem(h)) {
			fprint(2, "%s: mangled input file\n", pn);
			errorexit();
		}
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

	p = mal(sizeof(Prog));
	p->as = o;
	p->scond = BGETC(f);
	p->reg = BGETC(f);
	p->line = Bget4(f);

	zaddr(pn, f, &p->from, h);
	fromgotype = adrgotype;
	zaddr(pn, f, &p->to, h);

	if(p->as != ATEXT && p->as != AGLOBL && p->reg > NREG)
		diag("register out of range %A %d", p->as, p->reg);

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
		if(cursym != nil && cursym->text)
			cursym->autom = curauto;
		curauto = 0;
		cursym = nil;
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
		if(s->type != SBSS && s->type != SNOPTRBSS && !s->dupok) {
			diag("redefinition: %s\n%P", s->name, p);
			s->type = SBSS;
			s->value = 0;
		}
		if(p->to.offset > s->size)
			s->size = p->to.offset;
		if(p->reg & DUPOK)
			s->dupok = 1;
		if(p->reg & RODATA)
			s->type = SRODATA;
		else if(p->reg & NOPTR)
			s->type = SNOPTRBSS;
		break;

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
		break;

	case AGOK:
		diag("unknown opcode\n%P", p);
		p->pc = pc;
		pc++;
		break;

	case ALOCALS:
		if(skip)
			goto casedef;
		cursym->locals = p->to.offset;
		pc++;
		break;

	case ATYPE:
		if(skip)
			goto casedef;
		pc++;
		goto loop;

	case ATEXT:
		if(cursym != nil && cursym->text) {
			histtoauto();
			cursym->autom = curauto;
			curauto = 0;
		}
		s = p->from.sym;
		if(s == S) {
			diag("TEXT must have a name\n%P", p);
			errorexit();
		}
		cursym = s;
		if(s->type != 0 && s->type != SXREF && (p->reg & DUPOK)) {
			skip = 1;
			goto casedef;
		}
		if(ntext++ == 0 && s->type != 0 && s->type != SXREF) {
			/* redefinition, so file has probably been seen before */
			if(debug['v'])
				Bprint(&bso, "skipping: %s: redefinition: %s", pn, s->name);
			return;
		}
		skip = 0;
		if(s->type != 0 && s->type != SXREF)
			diag("redefinition: %s\n%P", s->name, p);
		if(etextp)
			etextp->next = s;
		else
			textp = s;
		if(fromgotype) {
			if(s->gotype && s->gotype != fromgotype)
				diag("%s: type mismatch for %s", pn, s->name);
			s->gotype = fromgotype;
		}
		etextp = s;
		p->align = 4;
		autosize = (p->to.offset+3L) & ~3L;
		p->to.offset = autosize;
		autosize += 4;
		s->type = STEXT;
		s->text = p;
		s->value = pc;
		s->args = p->to.offset2;
		lastp = p;
		p->pc = pc;
		pc++;
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
		goto casedef;

	case AMOVF:
		if(skip)
			goto casedef;

		if(p->from.type == D_FCONST && chipfloat(&p->from.ieee) < 0 &&
		   (chipzero(&p->from.ieee) < 0 || (p->scond & C_SCOND) != C_SCOND_NONE)) {
			/* size sb 9 max */
			sprint(literal, "$%ux", ieeedtof(&p->from.ieee));
			s = lookup(literal, 0);
			if(s->type == 0) {
				s->type = SRODATA;
				adduint32(s, ieeedtof(&p->from.ieee));
				s->reachable = 0;
			}
			p->from.type = D_OREG;
			p->from.sym = s;
			p->from.name = D_EXTERN;
			p->from.offset = 0;
		}
		goto casedef;

	case AMOVD:
		if(skip)
			goto casedef;

		if(p->from.type == D_FCONST && chipfloat(&p->from.ieee) < 0 &&
		   (chipzero(&p->from.ieee) < 0 || (p->scond & C_SCOND) != C_SCOND_NONE)) {
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
		p->pc = pc;
		pc++;
		if(p->to.type == D_BRANCH)
			p->to.offset += ipc;
		if(lastp == nil) {
			if(p->as != ANOP)
				diag("unexpected instruction: %P", p);
			break;
		}
		lastp->link = p;
		lastp = p;
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

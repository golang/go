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
#include	"../ld/elf.h"
#include	"../ld/macho.h"
#include	<ar.h>

char	*noname		= "<none>";
char	symname[]	= SYMDEF;
char	thechar		= '6';
char*	thestring 	= "amd64";
char*	paramspace	= "FP";

/*
 *	-H2 -T4136 -R4096		is plan9 64-bit format
 *	-H3 -T4128 -R4096		is plan9 32-bit format
 *	-H5 -T0x80110000 -R4096		is ELF32
 *	-H6 -Tx -Rx			is apple MH-exec
 *	-H7 -Tx -Rx			is linux elf-exec
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
	fprint(2, "usage: 6l [-options] [-E entry] [-H head] [-L dir] [-T text] [-R rnd] [-o out] files...\n");
	exits("usage");
}

void
main(int argc, char *argv[])
{
	int i, c;
	char *a;

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
	LIBDIR = nil;

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
		LIBDIR = EARGF(usage());
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

	mywhatsys();	// get goroot, goarch, goos
	if(strcmp(goarch, thestring) != 0)
		print("goarch is not known: %s\n", goarch);

	if(HEADTYPE == -1) {
		HEADTYPE = 2;
		if(strcmp(goos, "linux") == 0)
			HEADTYPE = 7;
		else
		if(strcmp(goos, "darwin") == 0)
			HEADTYPE = 6;
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
		if(INITTEXT == -1)
			INITTEXT = 4096+HEADR;
		if(INITDAT == -1)
			INITDAT = 0;
		if(INITRND == -1)
			INITRND = 4096;
		break;
	case 7:	/* elf64 executable */
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
	unlink(outfile);
	cout = create(outfile, 1, 0775);
	if(cout < 0) {
		diag("cannot create %s", outfile);
		errorexit();
	}
	version = 0;
	cbp = buf.cbuf;
	cbc = sizeof(buf.cbuf);
	firstp = prg();
	lastp = firstp;

	if(INITENTRY == nil) {
		INITENTRY = mal(strlen(goarch)+strlen(goos)+10);
		sprint(INITENTRY, "_rt0_%s_%s", goarch, goos);
	}
	lookup(INITENTRY, 0)->type = SXREF;

	while(*argv)
		objfile(*argv++);

	if(!debug['l']) {
		loadlib();
		a = mal(strlen(goroot)+strlen(goarch)+strlen(goos)+20);
		sprint(a, "%s/pkg/%s_%s/runtime.a", goroot, goos, goarch);
		objfile(a);
	}
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
		Bprint(&bso, "%ld memory used\n", thunk);
		Bprint(&bso, "%d sizeof adr\n", sizeof(Adr));
		Bprint(&bso, "%d sizeof prog\n", sizeof(Prog));
	}
	Bflush(&bso);

	errorexit();
}

void
loadlib(void)
{
	int i;
	int32 h;
	Sym *s;

loop:
	xrefresolv = 0;
	for(i=0; i<libraryp; i++) {
		if(debug['v'])
			Bprint(&bso, "%5.2f autolib: %s (from %s)\n", cputime(), library[i], libraryobj[i]);
		objfile(library[i]);
	}
	if(xrefresolv)
	for(h=0; h<nelem(hash); h++)
	for(s = hash[h]; s != S; s = s->link)
		if(s->type == SXREF)
			goto loop;
}

void
errorexit(void)
{
	if(nerrors) {
		if(cout >= 0)
			remove(outfile);
		exits("error");
	}
	exits(0);
}

void
objfile(char *file)
{
	int32 off, esym, cnt, l;
	int work;
	Biobuf *f;
	Sym *s;
	char magbuf[SARMAG];
	char name[100], pname[150];
	struct ar_hdr arhdr;
	char *e, *start, *stop;

	if(file[0] == '-' && file[1] == 'l') {	// TODO: fix this
		if(debug['9'])
			sprint(name, "/%s/lib/lib", thestring);
		else
			sprint(name, "/usr/%clib/lib", thechar);
		strcat(name, file+2);
		strcat(name, ".a");
		file = name;
	}
	if(debug['v'])
		Bprint(&bso, "%5.2f ldobj: %s\n", cputime(), file);
	Bflush(&bso);
	f = Bopen(file, 0);
	if(f == nil) {
		diag("cannot open file: %s", file);
		errorexit();
	}
	l = Bread(f, magbuf, SARMAG);
	if(l != SARMAG || strncmp(magbuf, ARMAG, SARMAG)){
		/* load it as a regular file */
		l = Bseek(f, 0L, 2);
		Bseek(f, 0L, 0);
		ldobj(f, l, file);
		Bterm(f);
		return;
	}

	l = Bread(f, &arhdr, SAR_HDR);
	if(l != SAR_HDR) {
		diag("%s: short read on archive file symbol header", file);
		goto out;
	}
	if(strncmp(arhdr.name, symname, strlen(symname))) {
		diag("%s: first entry not symbol header", file);
		goto out;
	}

	esym = SARMAG + SAR_HDR + atolwhex(arhdr.size);
	off = SARMAG + SAR_HDR;

	/*
	 * just bang the whole symbol file into memory
	 */
	Bseek(f, off, 0);
	cnt = esym - off;
	start = mal(cnt + 10);
	cnt = Bread(f, start, cnt);
	if(cnt <= 0){
		Bterm(f);
		return;
	}
	stop = &start[cnt];
	memset(stop, 0, 10);

	work = 1;
	while(work) {
		if(debug['v'])
			Bprint(&bso, "%5.2f library pass: %s\n", cputime(), file);
		Bflush(&bso);
		work = 0;
		for(e = start; e < stop; e = strchr(e+5, 0) + 1) {
			s = lookup(e+5, 0);
			if(s->type != SXREF)
				continue;
			sprint(pname, "%s(%s)", file, s->name);
			if(debug['v'])
				Bprint(&bso, "%5.2f library: %s\n", cputime(), pname);
			Bflush(&bso);
			l = e[1] & 0xff;
			l |= (e[2] & 0xff) << 8;
			l |= (e[3] & 0xff) << 16;
			l |= (e[4] & 0xff) << 24;
			Bseek(f, l, 0);
			l = Bread(f, &arhdr, SAR_HDR);
			if(l != SAR_HDR)
				goto bad;
			if(strncmp(arhdr.fmag, ARFMAG, sizeof(arhdr.fmag)))
				goto bad;
			l = SARNAME;
			while(l > 0 && arhdr.name[l-1] == ' ')
				l--;
			sprint(pname, "%s(%.*s)", file, l, arhdr.name);
			l = atolwhex(arhdr.size);
			ldobj(f, l, pname);
			if(s->type == SXREF) {
				diag("%s: failed to load: %s", file, s->name);
				errorexit();
			}
			work = 1;
			xrefresolv = 1;
		}
	}
	return;

bad:
	diag("%s: bad or out of date archive", file);
out:
	Bterm(f);
}

int32
Bget4(Biobuf *f)
{
	uchar p[4];

	if(Bread(f, p, 4) != 4)
		return 0;
	return p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24);
}

void
zaddr(Biobuf *f, Adr *a, Sym *h[])
{
	int t;
	int32 l;
	Sym *s;
	Auto *u;

	t = Bgetc(f);
	if(t & T_INDEX) {
		a->index = Bgetc(f);
		a->scale = Bgetc(f);
	} else {
		a->index = D_NONE;
		a->scale = 0;
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
	if(t & T_GOTYPE)
		a->gotype = h[Bgetc(f)];
	s = a->sym;
	if(s == S)
		return;

	t = a->type;
	if(t != D_AUTO && t != D_PARAM) {
		if(a->gotype)
			s->gotype = a->gotype;
		return;
	}
	l = a->offset;
	for(u=curauto; u; u=u->link) {
		if(u->asym == s)
		if(u->type == t) {
			if(u->aoffset > l)
				u->aoffset = l;
			if(a->gotype)
				u->gotype = a->gotype;
			return;
		}
	}

	u = mal(sizeof(*u));
	u->link = curauto;
	curauto = u;
	u->asym = s;
	u->aoffset = l;
	u->type = t;
	u->gotype = a->gotype;
}

void
addlib(char *src, char *obj)
{
	char name[1024], pname[1024], comp[256], *p;
	int i, search;

	if(histfrogp <= 0)
		return;

	search = 0;
	if(histfrog[0]->name[1] == '/') {
		sprint(name, "");
		i = 1;
	} else
	if(histfrog[0]->name[1] == '.') {
		sprint(name, ".");
		i = 0;
	} else {
		sprint(name, "");
		i = 0;
		search = 1;
	}

	for(; i<histfrogp; i++) {
		snprint(comp, sizeof comp, histfrog[i]->name+1);
		for(;;) {
			p = strstr(comp, "$O");
			if(p == 0)
				break;
			memmove(p+1, p+2, strlen(p+2)+1);
			p[0] = thechar;
		}
		for(;;) {
			p = strstr(comp, "$M");
			if(p == 0)
				break;
			if(strlen(comp)+strlen(thestring)-2+1 >= sizeof comp) {
				diag("library component too long");
				return;
			}
			memmove(p+strlen(thestring), p+2, strlen(p+2)+1);
			memmove(p, thestring, strlen(thestring));
		}
		if(strlen(name) + strlen(comp) + 3 >= sizeof(name)) {
			diag("library component too long");
			return;
		}
		strcat(name, "/");
		strcat(name, comp);
	}

	if(search) {
		// try dot, -L "libdir", and then goroot.
		snprint(pname, sizeof pname, "./%s", name);
		if(access(pname, AEXIST) < 0 && LIBDIR != nil)
			snprint(pname, sizeof pname, "%s/%s", LIBDIR, name);
		if(access(pname, AEXIST) < 0)
			snprint(pname, sizeof pname, "%s/pkg/%s_%s/%s", goroot, goos, goarch, name);
		strcpy(name, pname);
	}
	cleanname(name);
	if(debug['v'])
		Bprint(&bso, "%5.2f addlib: %s %s pulls in %s\n", cputime(), obj, src, name);

	for(i=0; i<libraryp; i++)
		if(strcmp(name, library[i]) == 0)
			return;
	if(libraryp == nelem(library)){
		diag("too many autolibs; skipping %s", name);
		return;
	}

	p = mal(strlen(name) + 1);
	strcpy(p, name);
	library[libraryp] = p;
	p = mal(strlen(obj) + 1);
	strcpy(p, obj);
	libraryobj[libraryp] = p;
	libraryp++;
}

void
copyhistfrog(char *buf, int nbuf)
{
	char *p, *ep;
	int i;

	p = buf;
	ep = buf + nbuf;
	i = 0;
	for(i=0; i<histfrogp; i++) {
		p = seprint(p, ep, "%s", histfrog[i]->name+1);
		if(i+1<histfrogp && (p == buf || p[-1] != '/'))
			p = seprint(p, ep, "/");
	}
}

void
addhist(int32 line, int type)
{
	Auto *u;
	Sym *s;
	int i, j, k;

	u = mal(sizeof(Auto));
	s = mal(sizeof(Sym));
	s->name = mal(2*(histfrogp+1) + 1);

	u->asym = s;
	u->type = type;
	u->aoffset = line;
	u->link = curhist;
	curhist = u;

	s->name[0] = 0;
	j = 1;
	for(i=0; i<histfrogp; i++) {
		k = histfrog[i]->value;
		s->name[j+0] = k>>8;
		s->name[j+1] = k;
		j += 2;
	}
	s->name[j] = 0;
	s->name[j+1] = 0;
}

void
histtoauto(void)
{
	Auto *l;

	while(l = curhist) {
		curhist = l->link;
		l->link = curauto;
		curauto = l;
	}
}

void
collapsefrog(Sym *s)
{
	int i;

	/*
	 * bad encoding of path components only allows
	 * MAXHIST components. if there is an overflow,
	 * first try to collapse xxx/..
	 */
	for(i=1; i<histfrogp; i++)
		if(strcmp(histfrog[i]->name+1, "..") == 0) {
			memmove(histfrog+i-1, histfrog+i+1,
				(histfrogp-i-1)*sizeof(histfrog[0]));
			histfrogp--;
			goto out;
		}

	/*
	 * next try to collapse .
	 */
	for(i=0; i<histfrogp; i++)
		if(strcmp(histfrog[i]->name+1, ".") == 0) {
			memmove(histfrog+i, histfrog+i+1,
				(histfrogp-i-1)*sizeof(histfrog[0]));
			goto out;
		}

	/*
	 * last chance, just truncate from front
	 */
	memmove(histfrog+0, histfrog+1,
		(histfrogp-1)*sizeof(histfrog[0]));

out:
	histfrog[histfrogp-1] = s;
}

void
nopout(Prog *p)
{
	p->as = ANOP;
	p->from.type = D_NONE;
	p->to.type = D_NONE;
}

void
ldobj(Biobuf *f, int64 len, char *pn)
{
	vlong ipc;
	Prog *p, *t;
	int v, o, r, skip, mode;
	Sym *h[NSYM], *s, *di;
	uint32 sig;
	static int files;
	static char **filen;
	char **nfilen, *line, *name;
	int ntext, n, c1, c2, c3;
	vlong eof;
	vlong import0, import1;
	char src[1024];

	src[0] = '\0';
	eof = Boffset(f) + len;

	ntext = 0;

	if((files&15) == 0){
		nfilen = malloc((files+16)*sizeof(char*));
		memmove(nfilen, filen, files*sizeof(char*));
		free(filen);
		filen = nfilen;
	}
	pn = strdup(pn);
	filen[files++] = pn;

	di = S;

	/* check the header */
	line = Brdline(f, '\n');
	if(line == nil) {
		if(Blinelen(f) > 0) {
			diag("%s: malformed object file", pn);
			return;
		}
		goto eof;
	}
	n = Blinelen(f) - 1;
	if(n != strlen(thestring) || strncmp(line, thestring, n) != 0) {
		if(line)
			line[n] = '\0';
		diag("file not %s [%s]\n", thestring, line);
		return;
	}

	/* skip over exports and other info -- ends with \n!\n */
	import0 = Boffset(f);
	c1 = '\n';	// the last line ended in \n
	c2 = Bgetc(f);
	c3 = Bgetc(f);
	while(c1 != '\n' || c2 != '!' || c3 != '\n') {
		c1 = c2;
		c2 = c3;
		c3 = Bgetc(f);
		if(c3 == Beof)
			goto eof;
	}
	import1 = Boffset(f);

	Bseek(f, import0, 0);
	ldpkg(f, import1 - import0 - 2, pn);	// -2 for !\n
	Bseek(f, import1, 0);

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
		s = lookup(name, r);

		if(debug['S'] && r == 0)
			sig = 1729;
		if(sig != 0){
			if(s->sig != 0 && s->sig != sig)
				diag("incompatible type signatures"
					"%lux(%s) and %lux(%s) for %s",
					s->sig, filen[s->file], sig, pn, s->name);
			s->sig = sig;
			s->file = files-1;
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
	zaddr(f, &p->from, h);
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
		if(p->from.gotype) {
			if(s->gotype && s->gotype != p->from.gotype)
				diag("%s: type mismatch for %s", pn, s->name);
			s->gotype = p->from.gotype;
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

Sym*
lookup(char *symb, int v)
{
	Sym *s;
	char *p;
	int32 h;
	int l, c;

	h = v;
	for(p=symb; c = *p; p++)
		h = h+h+h + c;
	l = (p - symb) + 1;
	if(h < 0)
		h = ~h;
	h %= NHASH;
	for(s = hash[h]; s != S; s = s->link)
		if(s->version == v)
		if(memcmp(s->name, symb, l) == 0)
			return s;

	s = mal(sizeof(*s));
	if(debug['v'] > 1)
		Bprint(&bso, "lookup %s\n", symb);

	s->name = mal(l + 1);
	memmove(s->name, symb, l);

	s->link = hash[h];
	s->type = 0;
	s->version = v;
	s->value = 0;
	s->sig = 0;
	hash[h] = s;
	nsymbol++;
	return s;
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

void
nuxiinit(void)
{
	int i, c;

	for(i=0; i<4; i++) {
		c = find1(0x04030201L, i+1);
		if(i < 2)
			inuxi2[i] = c;
		if(i < 1)
			inuxi1[i] = c;
		inuxi4[i] = c;
		inuxi8[i] = c;
		inuxi8[i+4] = c+4;
		fnuxi4[i] = c;
		fnuxi8[i] = c;
		fnuxi8[i+4] = c+4;
	}
	if(debug['v']) {
		Bprint(&bso, "inuxi = ");
		for(i=0; i<1; i++)
			Bprint(&bso, "%d", inuxi1[i]);
		Bprint(&bso, " ");
		for(i=0; i<2; i++)
			Bprint(&bso, "%d", inuxi2[i]);
		Bprint(&bso, " ");
		for(i=0; i<4; i++)
			Bprint(&bso, "%d", inuxi4[i]);
		Bprint(&bso, " ");
		for(i=0; i<8; i++)
			Bprint(&bso, "%d", inuxi8[i]);
		Bprint(&bso, "\nfnuxi = ");
		for(i=0; i<4; i++)
			Bprint(&bso, "%d", fnuxi4[i]);
		Bprint(&bso, " ");
		for(i=0; i<8; i++)
			Bprint(&bso, "%d", fnuxi8[i]);
		Bprint(&bso, "\n");
	}
	Bflush(&bso);
}

int
find1(int32 l, int c)
{
	char *p;
	int i;

	p = (char*)&l;
	for(i=0; i<4; i++)
		if(*p++ == c)
			return i;
	return 0;
}

int
find2(int32 l, int c)
{
	short *p;
	int i;

	p = (short*)&l;
	for(i=0; i<4; i+=2) {
		if(((*p >> 8) & 0xff) == c)
			return i;
		if((*p++ & 0xff) == c)
			return i+1;
	}
	return 0;
}

int32
ieeedtof(Ieee *e)
{
	int exp;
	int32 v;

	if(e->h == 0)
		return 0;
	exp = (e->h>>20) & ((1L<<11)-1L);
	exp -= (1L<<10) - 2L;
	v = (e->h & 0xfffffL) << 3;
	v |= (e->l >> 29) & 0x7L;
	if((e->l >> 28) & 1) {
		v++;
		if(v & 0x800000L) {
			v = (v & 0x7fffffL) >> 1;
			exp++;
		}
	}
	if(exp <= -126 || exp >= 130)
		diag("double fp to single fp overflow");
	v |= ((exp + 126) & 0xffL) << 23;
	v |= e->h & 0x80000000L;
	return v;
}

double
ieeedtod(Ieee *ieeep)
{
	Ieee e;
	double fr;
	int exp;

	if(ieeep->h & (1L<<31)) {
		e.h = ieeep->h & ~(1L<<31);
		e.l = ieeep->l;
		return -ieeedtod(&e);
	}
	if(ieeep->l == 0 && ieeep->h == 0)
		return 0;
	fr = ieeep->l & ((1L<<16)-1L);
	fr /= 1L<<16;
	fr += (ieeep->l>>16) & ((1L<<16)-1L);
	fr /= 1L<<16;
	fr += (ieeep->h & (1L<<20)-1L) | (1L<<20);
	fr /= 1L<<21;
	exp = (ieeep->h>>20) & ((1L<<11)-1L);
	exp -= (1L<<10) - 2L;
	return ldexp(fr, exp);
}

void
undefsym(Sym *s)
{
	int n;

	n = imports;
	if(s->value != 0)
		diag("value != 0 on SXREF");
	if(n >= 1<<Rindex)
		diag("import index %d out of range", n);
	s->value = n<<Roffset;
	s->type = SUNDEF;
	imports++;
}

void
zerosig(char *sp)
{
	Sym *s;

	s = lookup(sp, 0);
	s->sig = 0;
}

void
readundefs(char *f, int t)
{
	int i, n;
	Sym *s;
	Biobuf *b;
	char *l, buf[256], *fields[64];

	if(f == nil)
		return;
	b = Bopen(f, OREAD);
	if(b == nil){
		diag("could not open %s: %r", f);
		errorexit();
	}
	while((l = Brdline(b, '\n')) != nil){
		n = Blinelen(b);
		if(n >= sizeof(buf)){
			diag("%s: line too long", f);
			errorexit();
		}
		memmove(buf, l, n);
		buf[n-1] = '\0';
		n = getfields(buf, fields, nelem(fields), 1, " \t\r\n");
		if(n == nelem(fields)){
			diag("%s: bad format", f);
			errorexit();
		}
		for(i = 0; i < n; i++){
			s = lookup(fields[i], 0);
			s->type = SXREF;
			s->subtype = t;
			if(t == SIMPORT)
				nimports++;
			else
				nexports++;
		}
	}
	Bterm(b);
}

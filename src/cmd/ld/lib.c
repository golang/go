// Derived from Inferno utils/6l/obj.c and utils/6l/span.c
// http://code.google.com/p/inferno-os/source/browse/utils/6l/obj.c
// http://code.google.com/p/inferno-os/source/browse/utils/6l/span.c
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

#include	"l.h"
#include	"lib.h"
#include	<ar.h>

int iconv(Fmt*);

char	symname[]	= SYMDEF;
char	pkgname[]	= "__.PKGDEF";
char*	libdir[16];
int	nlibdir = 0;
int	cout = -1;

char*	goroot;
char*	goarch;
char*	goos;

void
Lflag(char *arg)
{
	if(nlibdir >= nelem(libdir)-1) {
		print("too many -L's: %d\n", nlibdir);
		usage();
	}
	libdir[nlibdir++] = arg;
}

void
libinit(void)
{
	fmtinstall('i', iconv);
	mywhatsys();	// get goroot, goarch, goos
	if(strcmp(goarch, thestring) != 0)
		print("goarch is not known: %s\n", goarch);

	// add goroot to the end of the libdir list.
	libdir[nlibdir++] = smprint("%s/pkg/%s_%s", goroot, goos, goarch);

	unlink(outfile);
	cout = create(outfile, 1, 0775);
	if(cout < 0) {
		diag("cannot create %s", outfile);
		errorexit();
	}

	if(INITENTRY == nil) {
		INITENTRY = mal(strlen(goarch)+strlen(goos)+10);
		sprint(INITENTRY, "_rt0_%s_%s", goarch, goos);
	}
	lookup(INITENTRY, 0)->type = SXREF;
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
	if(isalpha(histfrog[0]->name[1]) && histfrog[0]->name[2] == ':') {
		strcpy(name, histfrog[0]->name+1);
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
		snprint(comp, sizeof comp, "%s", histfrog[i]->name+1);
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
		if(i > 0 || !search)
			strcat(name, "/");
		strcat(name, comp);
	}
	cleanname(name);
	
	// runtime.a -> runtime
	p = nil;
	if(strlen(name) > 2 && name[strlen(name)-2] == '.') {
		p = name+strlen(name)-2;
		*p = '\0';
	}
	
	// already loaded?
	for(i=0; i<libraryp; i++)
		if(strcmp(library[i].pkg, name) == 0)
			return;
	
	// runtime -> runtime.a for search
	if(p != nil)
		*p = '.';

	if(search) {
		// try dot, -L "libdir", and then goroot.
		for(i=0; i<nlibdir; i++) {
			snprint(pname, sizeof pname, "%s/%s", libdir[i], name);
			if(access(pname, AEXIST) >= 0)
				break;
		}
	}else
		strcpy(pname, name);
	cleanname(pname);

	/* runtime.a -> runtime */
	if(p != nil)
		*p = '\0';

	if(debug['v'])
		Bprint(&bso, "%5.2f addlib: %s %s pulls in %s\n", cputime(), obj, src, pname);

	addlibpath(src, obj, pname, name);
}

/*
 * add library to library list.
 *	srcref: src file referring to package
 *	objref: object file referring to package
 *	file: object file, e.g., /home/rsc/go/pkg/container/vector.a
 *	pkg: package import path, e.g. container/vector
 */
void
addlibpath(char *srcref, char *objref, char *file, char *pkg)
{
	int i;
	Library *l;
	char *p;

	for(i=0; i<libraryp; i++)
		if(strcmp(file, library[i].file) == 0)
			return;

	if(debug['v'] > 1)
		Bprint(&bso, "%5.2f addlibpath: srcref: %s objref: %s file: %s pkg: %s\n",
			cputime(), srcref, objref, file, pkg);

	if(libraryp == nlibrary){
		nlibrary = 50 + 2*libraryp;
		library = realloc(library, sizeof library[0] * nlibrary);
	}

	l = &library[libraryp++];

	p = mal(strlen(objref) + 1);
	strcpy(p, objref);
	l->objref = p;

	p = mal(strlen(srcref) + 1);
	strcpy(p, srcref);
	l->srcref = p;

	p = mal(strlen(file) + 1);
	strcpy(p, file);
	l->file = p;

	p = mal(strlen(pkg) + 1);
	strcpy(p, pkg);
	l->pkg = p;
}

void
loadlib(void)
{
	char pname[1024];
	int i, found;

	found = 0;
	for(i=0; i<nlibdir; i++) {
		snprint(pname, sizeof pname, "%s/runtime.a", libdir[i]);
		if(debug['v'])
			Bprint(&bso, "searching for runtime.a in %s\n", pname);
		if(access(pname, AEXIST) >= 0) {
			addlibpath("internal", "internal", pname, "runtime");
			found = 1;
			break;
		}
	}
	if(!found)
		Bprint(&bso, "warning: unable to find runtime.a\n");

	for(i=0; i<libraryp; i++) {
		if(debug['v'])
			Bprint(&bso, "%5.2f autolib: %s (from %s)\n", cputime(), library[i].file, library[i].objref);
		objfile(library[i].file, library[i].pkg);
	}
}

/*
 * look for the next file in an archive.
 * adapted from libmach.
 */
int
nextar(Biobuf *bp, int off, struct ar_hdr *a)
{
	int r;
	int32 arsize;

	if (off&01)
		off++;
	Bseek(bp, off, 0);
	r = Bread(bp, a, SAR_HDR);
	if(r != SAR_HDR)
		return 0;
	if(strncmp(a->fmag, ARFMAG, sizeof(a->fmag)))
		return -1;
	arsize = strtol(a->size, 0, 0);
	if (arsize&1)
		arsize++;
	return arsize + SAR_HDR;
}

void
objfile(char *file, char *pkg)
{
	int32 off, l;
	Biobuf *f;
	char magbuf[SARMAG];
	char pname[150];
	struct ar_hdr arhdr;

	pkg = smprint("%i", pkg);

	if(debug['v'])
		Bprint(&bso, "%5.2f ldobj: %s (%s)\n", cputime(), file, pkg);
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
		ldobj(f, pkg, l, file, FileObj);
		Bterm(f);
		return;
	}
	
	/* skip over __.SYMDEF */
	off = Boffset(f);
	if((l = nextar(f, off, &arhdr)) <= 0) {
		diag("%s: short read on archive file symbol header", file);
		goto out;
	}
	if(strncmp(arhdr.name, symname, strlen(symname))) {
		diag("%s: first entry not symbol header", file);
		goto out;
	}
	off += l;
	
	/* skip over (or process) __.PKGDEF */
	if((l = nextar(f, off, &arhdr)) <= 0) {
		diag("%s: short read on archive file symbol header", file);
		goto out;
	}
	if(strncmp(arhdr.name, pkgname, strlen(pkgname))) {
		diag("%s: second entry not package header", file);
		goto out;
	}
	off += l;

	if(debug['u'])
		ldpkg(f, pkg, atolwhex(arhdr.size), file, Pkgdef);

	/*
	 * load all the object files from the archive now.
	 * this gives us sequential file access and keeps us
	 * from needing to come back later to pick up more
	 * objects.  it breaks the usual C archive model, but
	 * this is Go, not C.  the common case in Go is that
	 * we need to load all the objects, and then we throw away
	 * the individual symbols that are unused.
	 *
	 * loading every object will also make it possible to
	 * load foreign objects not referenced by __.SYMDEF.
	 */
	for(;;) {
		l = nextar(f, off, &arhdr);
		if(l == 0)
			break;
		if(l < 0) {
			diag("%s: malformed archive", file);
			goto out;
		}
		off += l;

		l = SARNAME;
		while(l > 0 && arhdr.name[l-1] == ' ')
			l--;
		snprint(pname, sizeof pname, "%s(%.*s)", file, utfnlen(arhdr.name, l), arhdr.name);
		l = atolwhex(arhdr.size);
		ldobj(f, pkg, l, pname, ArchiveObj);
	}

out:
	Bterm(f);
}

void
ldobj(Biobuf *f, char *pkg, int64 len, char *pn, int whence)
{
	char *line;
	int n, c1, c2, c3, c4;
	uint32 magic;
	vlong import0, import1, eof;
	char src[1024];

	eof = Boffset(f) + len;
	src[0] = '\0';

	pn = strdup(pn);
	
	USED(c4);
	USED(magic);

	c1 = Bgetc(f);
	c2 = Bgetc(f);
	c3 = Bgetc(f);
	c4 = Bgetc(f);
	Bungetc(f);
	Bungetc(f);
	Bungetc(f);
	Bungetc(f);
	
	magic = c1<<24 | c2<<16 | c3<<8 | c4;
	if(magic == 0x7f454c46) {	// \x7F E L F
		ldelf(f, pkg, len, pn);
		return;
	}
	if((magic&~1) == 0xfeedface || (magic&~0x01000000) == 0xcefaedfe) {
		ldmacho(f, pkg, len, pn);
		return;
	}

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
		if(strlen(pn) > 3 && strcmp(pn+strlen(pn)-3, ".go") == 0) {
			print("%cl: input %s is not .%c file (use %cg to compile .go files)\n", thechar, pn, thechar, thechar);
			errorexit();
		}
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
	ldpkg(f, pkg, import1 - import0 - 2, pn, whence);	// -2 for !\n
	Bseek(f, import1, 0);

	ldobj1(f, pkg, eof - Boffset(f), pn);
	return;

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
	// not if(h < 0) h = ~h, because gcc 4.3 -O2 miscompiles it.
	h &= 0xffffff;
	h %= NHASH;
	for(s = hash[h]; s != S; s = s->hash)
		if(s->version == v)
		if(memcmp(s->name, symb, l) == 0)
			return s;

	s = mal(sizeof(*s));
	if(debug['v'] > 1)
		Bprint(&bso, "lookup %s\n", symb);

	s->dynid = -1;
	s->plt = -1;
	s->got = -1;
	s->name = mal(l + 1);
	memmove(s->name, symb, l);

	s->hash = hash[h];
	s->type = 0;
	s->version = v;
	s->value = 0;
	s->sig = 0;
	s->size = 0;
	hash[h] = s;
	nsymbol++;
	return s;
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
		if(c == i) {
			inuxi8[i] = c;
			inuxi8[i+4] = c+4;
		} else {
			inuxi8[i] = c+4;
			inuxi8[i+4] = c;
		}
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
	union {
		int32 l;
		short p[2];
	} u;
	short *p;
	int i;

	u.l = l;
	p = u.p;
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
	if(-148 <= exp && exp <= -126) {
		v |= 1<<23;
		v >>= -125 - exp;
		exp = -126;
	}
	else if(exp < -148 || exp >= 130)
		diag("double fp to single fp overflow: %.17g", ieeedtod(e));
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
	exp = (ieeep->h>>20) & ((1L<<11)-1L);
	exp -= (1L<<10) - 2L;
	fr = ieeep->l & ((1L<<16)-1L);
	fr /= 1L<<16;
	fr += (ieeep->l>>16) & ((1L<<16)-1L);
	fr /= 1L<<16;
	if(exp == -(1L<<10) - 2L) {
		fr += (ieeep->h & (1L<<20)-1L);
		exp++;
	} else
		fr += (ieeep->h & (1L<<20)-1L) | (1L<<20);
	fr /= 1L<<21;
	return ldexp(fr, exp);
}

void
zerosig(char *sp)
{
	Sym *s;

	s = lookup(sp, 0);
	s->sig = 0;
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
mywhatsys(void)
{
	goroot = getgoroot();
	goos = getgoos();
	goarch = thestring;	// ignore $GOARCH - we know who we are
}

int
pathchar(void)
{
	return '/';
}

static	uchar*	hunk;
static	uint32	nhunk;
#define	NHUNK	(10UL<<20)

void*
mal(uint32 n)
{
	void *v;

	n = (n+7)&~7;
	if(n > NHUNK) {
		v = malloc(n);
		if(v == nil) {
			diag("out of memory");
			errorexit();
		}
		memset(v, 0, n);
		return v;
	}
	if(n > nhunk) {
		hunk = malloc(NHUNK);
		if(hunk == nil) {
			diag("out of memory");
			errorexit();
		}
		nhunk = NHUNK;
	}

	v = hunk;
	nhunk -= n;
	hunk += n;

	memset(v, 0, n);
	return v;
}

void
unmal(void *v, uint32 n)
{
	n = (n+7)&~7;
	if(hunk - n == v) {
		hunk -= n;
		nhunk += n;
	}
}

// Copied from ../gc/subr.c:/^pathtoprefix; must stay in sync.
/*
 * Convert raw string to the prefix that will be used in the symbol table.
 * Invalid bytes turn into %xx.  Right now the only bytes that need
 * escaping are %, ., and ", but we escape all control characters too.
 */
static char*
pathtoprefix(char *s)
{
	static char hex[] = "0123456789abcdef";
	char *p, *r, *w;
	int n;

	// check for chars that need escaping
	n = 0;
	for(r=s; *r; r++)
		if(*r <= ' ' || *r == '.' || *r == '%' || *r == '"')
			n++;

	// quick exit
	if(n == 0)
		return s;

	// escape
	p = mal((r-s)+1+2*n);
	for(r=s, w=p; *r; r++) {
		if(*r <= ' ' || *r == '.' || *r == '%' || *r == '"') {
			*w++ = '%';
			*w++ = hex[(*r>>4)&0xF];
			*w++ = hex[*r&0xF];
		} else
			*w++ = *r;
	}
	*w = '\0';
	return p;
}

int
iconv(Fmt *fp)
{
	char *p;

	p = va_arg(fp->args, char*);
	if(p == nil) {
		fmtstrcpy(fp, "<nil>");
		return 0;
	}
	p = pathtoprefix(p);
	fmtstrcpy(fp, p);
	return 0;
}

void
mangle(char *file)
{
	fprint(2, "%s: mangled input file\n", file);
	errorexit();
}

Section*
addsection(Segment *seg, char *name, int rwx)
{
	Section **l;
	Section *sect;
	
	for(l=&seg->sect; *l; l=&(*l)->next)
		;
	sect = mal(sizeof *sect);
	sect->rwx = rwx;
	sect->name = name;
	sect->seg = seg;
	*l = sect;
	return sect;
}

void
ewrite(int fd, void *buf, int n)
{
	if(write(fd, buf, n) < 0) {
		diag("write error: %r");
		errorexit();
	}
}

void
pclntab(void)
{
	vlong oldpc;
	Prog *p;
	int32 oldlc, v, s;
	Sym *sym;
	uchar *bp;
	
	sym = lookup("pclntab", 0);
	sym->type = SRODATA;
	sym->reachable = 1;
	if(debug['s'])
		return;

	oldpc = INITTEXT;
	oldlc = 0;
	for(cursym = textp; cursym != nil; cursym = cursym->next) {
		for(p = cursym->text; p != P; p = p->link) {
			if(p->line == oldlc || p->as == ATEXT || p->as == ANOP) {
				if(debug['O'])
					Bprint(&bso, "%6llux %P\n",
						p->pc, p);
				continue;
			}
			if(debug['O'])
				Bprint(&bso, "\t\t%6d", lcsize);
			v = (p->pc - oldpc) / MINLC;
			while(v) {
				s = 127;
				if(v < 127)
					s = v;
				symgrow(sym, lcsize+1);
				bp = sym->p + lcsize;
				*bp = s+128;	/* 129-255 +pc */
				if(debug['O'])
					Bprint(&bso, " pc+%d*%d(%d)", s, MINLC, s+128);
				v -= s;
				lcsize++;
			}
			s = p->line - oldlc;
			oldlc = p->line;
			oldpc = p->pc + MINLC;
			if(s > 64 || s < -64) {
				symgrow(sym, lcsize+5);
				bp = sym->p + lcsize;
				*bp++ = 0;	/* 0 vv +lc */
				*bp++ = s>>24;
				*bp++ = s>>16;
				*bp++ = s>>8;
				*bp = s;
				if(debug['O']) {
					if(s > 0)
						Bprint(&bso, " lc+%d(%d,%d)\n",
							s, 0, s);
					else
						Bprint(&bso, " lc%d(%d,%d)\n",
							s, 0, s);
					Bprint(&bso, "%6llux %P\n",
						p->pc, p);
				}
				lcsize += 5;
				continue;
			}
			symgrow(sym, lcsize+1);
			bp = sym->p + lcsize;
			if(s > 0) {
				*bp = 0+s;	/* 1-64 +lc */
				if(debug['O']) {
					Bprint(&bso, " lc+%d(%d)\n", s, 0+s);
					Bprint(&bso, "%6llux %P\n",
						p->pc, p);
				}
			} else {
				*bp = 64-s;	/* 65-128 -lc */
				if(debug['O']) {
					Bprint(&bso, " lc%d(%d)\n", s, 64-s);
					Bprint(&bso, "%6llux %P\n",
						p->pc, p);
				}
			}
			lcsize++;
		}
	}
	if(lcsize & 1) {
		symgrow(sym, lcsize+1);
		sym->p[lcsize] = 129;
		lcsize++;
	}
	sym->size = lcsize;
	lcsize = 0;

	if(debug['v'] || debug['O'])
		Bprint(&bso, "lcsize = %d\n", lcsize);
	Bflush(&bso);
}

#define	LOG	5
void
mkfwd(void)
{
	Prog *p;
	int i;
	int32 dwn[LOG], cnt[LOG];
	Prog *lst[LOG], *last;

	for(i=0; i<LOG; i++) {
		if(i == 0)
			cnt[i] = 1;
		else
			cnt[i] = LOG * cnt[i-1];
		dwn[i] = 1;
		lst[i] = P;
	}
	i = 0;
	last = nil;
	for(cursym = textp; cursym != nil; cursym = cursym->next) {
		for(p = cursym->text; p != P; p = p->link) {
			if(p->link == P) {
				if(cursym->next)
					p->forwd = cursym->next->text;
				break;
			}
			i--;
			if(i < 0)
				i = LOG-1;
			p->forwd = P;
			dwn[i]--;
			if(dwn[i] <= 0) {
				dwn[i] = cnt[i];
				if(lst[i] != P)
					lst[i]->forwd = p;
				lst[i] = p;
			}
		}
	}
}

uint16
le16(uchar *b)
{
	return b[0] | b[1]<<8;
}

uint32
le32(uchar *b)
{
	return b[0] | b[1]<<8 | b[2]<<16 | b[3]<<24;
}

uint64
le64(uchar *b)
{
	return le32(b) | (uint64)le32(b+4)<<32;
}

uint16
be16(uchar *b)
{
	return b[0]<<8 | b[1];
}

uint32
be32(uchar *b)
{
	return b[0]<<24 | b[1]<<16 | b[2]<<8 | b[3];
}

uint64
be64(uchar *b)
{
	return (uvlong)be32(b)<<32 | be32(b+4);
}

Endian be = { be16, be32, be64 };
Endian le = { le16, le32, le64 };

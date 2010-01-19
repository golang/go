// Derived from Inferno utils/6l/obj.c
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

#include	"l.h"
#include	"lib.h"
#include	<ar.h>

char	symname[]	= SYMDEF;
char*	libdir[16] = { "." };
int	nlibdir = 1;
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
		for(i=0; i<nlibdir; i++) {
			snprint(pname, sizeof pname, "%s/%s", libdir[i], name);
			if(access(pname, AEXIST) >= 0)
				break;
		}
		strcpy(name, pname);
	}
	cleanname(name);
	if(debug['v'])
		Bprint(&bso, "%5.2f addlib: %s %s pulls in %s\n", cputime(), obj, src, name);

	for(i=0; i<libraryp; i++)
		if(strcmp(name, library[i]) == 0)
			return;
	if(libraryp == nlibrary){
		nlibrary = 50 + 2*libraryp;
		library = realloc(library, sizeof library[0] * nlibrary);
		libraryobj = realloc(libraryobj, sizeof libraryobj[0] * nlibrary);
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
loadlib(void)
{
	int i;
	int32 h;
	Sym *s;
	char *a;

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

	i = strlen(goroot)+strlen(goarch)+strlen(goos)+20;
	a = mal(i);
	snprint(a, i, "%s/pkg/%s_%s/runtime.a", goroot, goos, goarch);
	objfile(a);
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

void
ldobj(Biobuf *f, int64 len, char *pn)
{
	static int files;
	static char **filen;
	char **nfilen, *line;
	int n, c1, c2, c3;
	vlong import0, import1, eof;
	char src[1024];

	eof = Boffset(f) + len;
	src[0] = '\0';

	if((files&15) == 0){
		nfilen = malloc((files+16)*sizeof(char*));
		memmove(nfilen, filen, files*sizeof(char*));
		free(filen);
		filen = nfilen;
	}
	pn = strdup(pn);
	filen[files++] = pn;


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

	ldobj1(f, eof - Boffset(f), pn);
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

	while(n & 7)
		n++;
	if(n > NHUNK) {
		v = malloc(n);
		memset(v, 0, n);
		return v;
	}
	if(n > nhunk) {
		hunk = malloc(NHUNK);
		nhunk = NHUNK;
	}

	v = hunk;
	nhunk -= n;
	hunk += n;

	memset(v, 0, n);
	return v;
}

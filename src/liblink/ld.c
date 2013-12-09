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

#include <u.h>
#include <libc.h>
#include <bio.h>
#include <link.h>

void
copyhistfrog(Link *ctxt, char *buf, int nbuf)
{
	char *p, *ep;
	int i;

	p = buf;
	ep = buf + nbuf;
	for(i=0; i<ctxt->histfrogp; i++) {
		p = seprint(p, ep, "%s", ctxt->histfrog[i]->name+1);
		if(i+1<ctxt->histfrogp && (p == buf || p[-1] != '/'))
			p = seprint(p, ep, "/");
	}
}

void
addhist(Link *ctxt, int32 line, int type)
{
	Auto *u;
	LSym *s;
	int i, j, k;

	u = emallocz(sizeof(Auto));
	s = emallocz(sizeof(LSym));
	s->name = emallocz(2*(ctxt->histfrogp+1) + 1);

	u->asym = s;
	u->type = type;
	u->aoffset = line;
	u->link = ctxt->curhist;
	ctxt->curhist = u;

	s->name[0] = 0;
	j = 1;
	for(i=0; i<ctxt->histfrogp; i++) {
		k = ctxt->histfrog[i]->value;
		s->name[j+0] = k>>8;
		s->name[j+1] = k;
		j += 2;
	}
	s->name[j] = 0;
	s->name[j+1] = 0;
}

void
histtoauto(Link *ctxt)
{
	Auto *l;

	while(l = ctxt->curhist) {
		ctxt->curhist = l->link;
		l->link = ctxt->curauto;
		ctxt->curauto = l;
	}
}

void
collapsefrog(Link *ctxt, LSym *s)
{
	int i;

	/*
	 * bad encoding of path components only allows
	 * MAXHIST components. if there is an overflow,
	 * first try to collapse xxx/..
	 */
	for(i=1; i<ctxt->histfrogp; i++)
		if(strcmp(ctxt->histfrog[i]->name+1, "..") == 0) {
			memmove(ctxt->histfrog+i-1, ctxt->histfrog+i+1,
				(ctxt->histfrogp-i-1)*sizeof(ctxt->histfrog[0]));
			ctxt->histfrogp--;
			goto out;
		}

	/*
	 * next try to collapse .
	 */
	for(i=0; i<ctxt->histfrogp; i++)
		if(strcmp(ctxt->histfrog[i]->name+1, ".") == 0) {
			memmove(ctxt->histfrog+i, ctxt->histfrog+i+1,
				(ctxt->histfrogp-i-1)*sizeof(ctxt->histfrog[0]));
			goto out;
		}

	/*
	 * last chance, just truncate from front
	 */
	memmove(ctxt->histfrog+0, ctxt->histfrog+1,
		(ctxt->histfrogp-1)*sizeof(ctxt->histfrog[0]));

out:
	ctxt->histfrog[ctxt->histfrogp-1] = s;
}

// Saved history stacks encountered while reading archives.
// Keeping them allows us to answer virtual lineno -> file:line
// queries.
//
// The history stack is a complex data structure, described best at the
// bottom of http://plan9.bell-labs.com/magic/man2html/6/a.out.
// One of the key benefits of interpreting it here is that the runtime
// does not have to. Perhaps some day the compilers could generate
// a simpler linker input too.

// savehist processes a single line, off history directive
// found in the input object file.
void
savehist(Link *ctxt, int32 line, int32 off)
{
	char tmp[1024];
	LSym *file;
	Hist2 *h;

	// NOTE(rsc): We used to do the copyctxt->histfrog first and this
	// condition was if(tmp[0] != '\0') to check for an empty string,
	// implying that ctxt->histfrogp == 0, implying that this is a history pop.
	// However, on Windows in the misc/cgo test, the linker is
	// presented with an ANAME corresponding to an empty string,
	// that ANAME ends up being the only ctxt->histfrog, and thus we have
	// a situation where ctxt->histfrogp > 0 (not a pop) but the path we find
	// is the empty string. Really that shouldn't happen, but it doesn't
	// seem to be bothering anyone yet, and it's easier to fix the condition
	// to test ctxt->histfrogp than to track down where that empty string is
	// coming from. Probably it is coming from go tool pack's P command.
	if(ctxt->histfrogp > 0) {
		tmp[0] = '\0';
		copyhistfrog(ctxt, tmp, sizeof tmp);
		file = linklookup(ctxt, tmp, HistVersion);
	} else
		file = nil;

	if(file != nil && line == 1 && off == 0) {
		// start of new stack
		if(ctxt->histdepth != 0)
			sysfatal("history stack phase error: unexpected start of new stack depth=%d file=%s", ctxt->histdepth, tmp);
		ctxt->nhist2 = 0;
		ctxt->histcopy = nil;
	}
	
	if(ctxt->nhist2 >= ctxt->maxhist2) {
		if(ctxt->maxhist2 == 0)
			ctxt->maxhist2 = 1;
		ctxt->maxhist2 *= 2;
		ctxt->hist2 = erealloc(ctxt->hist2, ctxt->maxhist2*sizeof ctxt->hist2[0]);
	}
	h = &ctxt->hist2[ctxt->nhist2++];
	h->line = line;
	h->off = off;
	h->file = file;
	
	if(file != nil) {
		if(off == 0)
			ctxt->histdepth++;
	} else {
		if(off != 0)
			sysfatal("history stack phase error: bad offset in pop");
		ctxt->histdepth--;
	}
}

// gethist returns the history stack currently in effect.
// The result is valid indefinitely.
Hist2*
gethist(Link *ctxt)
{
	if(ctxt->histcopy == nil) {
		if(ctxt->nhist2 == 0)
			return nil;
		ctxt->histcopy = emallocz((ctxt->nhist2+1)*sizeof ctxt->hist2[0]);
		memmove(ctxt->histcopy, ctxt->hist2, ctxt->nhist2*sizeof ctxt->hist2[0]);
		ctxt->histcopy[ctxt->nhist2].line = -1;
	}
	return ctxt->histcopy;
}

typedef struct Hstack Hstack;
struct Hstack
{
	Hist2 *h;
	int delta;
};

// getline sets *f to the file number and *l to the line number
// of the virtual line number line according to the history stack h.
void
linkgetline(Link *ctxt, Hist2 *h, int32 line, LSym **f, int32 *l)
{
	Hstack stk[100];
	int nstk, start;
	Hist2 *top, *h0;
	static Hist2 *lasth;
	static int32 laststart, lastend, lastdelta;
	static LSym *lastfile;

	h0 = h;
	*f = 0;
	*l = 0;
	start = 0;
	if(h == nil || line == 0) {
		print("%s: getline: h=%p line=%d\n", ctxt->cursym->name, h, line);
		return;
	}

	// Cache span used during last lookup, so that sequential
	// translation of line numbers in compiled code is efficient.
	if(!ctxt->debughist && lasth == h && laststart <= line && line < lastend) {
		*f = lastfile;
		*l = line - lastdelta;
		return;
	}

	if(ctxt->debughist)
		print("getline %d laststart=%d lastend=%d\n", line, laststart, lastend);
	
	nstk = 0;
	for(; h->line != -1; h++) {
		if(ctxt->debughist)
			print("\t%s %d %d\n", h->file ? h->file->name : "?", h->line, h->off);

		if(h->line > line) {
			if(nstk == 0)
				sysfatal("history stack phase error: empty stack at line %d", (int)line);
			top = stk[nstk-1].h;
			lasth = h;
			lastfile = top->file;
			laststart = start;
			lastend = h->line;
			lastdelta = stk[nstk-1].delta;
			*f = lastfile;
			*l = line - lastdelta;
			if(ctxt->debughist)
				print("\tgot %d %d [%d %d %d]\n", *f, *l, laststart, lastend, lastdelta);
			return;
		}
		if(h->file == nil) {
			// pop included file
			if(nstk == 0)
				sysfatal("history stack phase error: stack underflow");
			nstk--;
			if(nstk > 0)
				stk[nstk-1].delta += h->line - stk[nstk].h->line;
			start = h->line;
		} else if(h->off == 0) {
			// push included file
			if(nstk >= nelem(stk))
				sysfatal("history stack phase error: stack overflow");
			start = h->line;
			stk[nstk].h = h;
			stk[nstk].delta = h->line - 1;
			nstk++;
		} else {
			// #line directive
			if(nstk == 0)
				sysfatal("history stack phase error: stack underflow");
			stk[nstk-1].h = h;
			stk[nstk-1].delta = h->line - h->off;
			start = h->line;
		}
		if(ctxt->debughist)
			print("\t\tnstk=%d delta=%d\n", nstk, stk[nstk].delta);
	}

	sysfatal("history stack phase error: cannot find line for %d", line);
	nstk = 0;
	for(h = h0; h->line != -1; h++) {
		print("\t%d %d %s\n", h->line, h->off, h->file ? h->file->name : "");
		if(h->file == nil)
			nstk--;
		else if(h->off == 0)
			nstk++;
	}
}

void
addlib(Link *ctxt, char *src, char *obj)
{
	char name[1024], pname[1024], comp[256], *p;
	int i, search;

	if(ctxt->histfrogp <= 0)
		return;

	search = 0;
	if(ctxt->histfrog[0]->name[1] == '/') {
		sprint(name, "");
		i = 1;
	} else
	if(isalpha((uchar)ctxt->histfrog[0]->name[1]) && ctxt->histfrog[0]->name[2] == ':') {
		strcpy(name, ctxt->histfrog[0]->name+1);
		i = 1;
	} else
	if(ctxt->histfrog[0]->name[1] == '.') {
		sprint(name, ".");
		i = 0;
	} else {
		sprint(name, "");
		i = 0;
		search = 1;
	}

	for(; i<ctxt->histfrogp; i++) {
		snprint(comp, sizeof comp, "%s", ctxt->histfrog[i]->name+1);
		for(;;) {
			p = strstr(comp, "$O");
			if(p == 0)
				break;
			memmove(p+1, p+2, strlen(p+2)+1);
			p[0] = ctxt->thechar;
		}
		for(;;) {
			p = strstr(comp, "$M");
			if(p == 0)
				break;
			if(strlen(comp)+strlen(ctxt->thestring)-2+1 >= sizeof comp)
				sysfatal("library component too long");
			memmove(p+strlen(ctxt->thestring), p+2, strlen(p+2)+1);
			memmove(p, ctxt->thestring, strlen(ctxt->thestring));
		}
		if(strlen(name) + strlen(comp) + 3 >= sizeof(name))
			sysfatal("library component too long");
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
	for(i=0; i<ctxt->libraryp; i++)
		if(strcmp(ctxt->library[i].pkg, name) == 0)
			return;
	
	// runtime -> runtime.a for search
	if(p != nil)
		*p = '.';

	if(search) {
		// try dot, -L "libdir", and then goroot.
		for(i=0; i<ctxt->nlibdir; i++) {
			snprint(pname, sizeof pname, "%s/%s", ctxt->libdir[i], name);
			if(access(pname, AEXIST) >= 0)
				break;
		}
	}else
		strcpy(pname, name);
	cleanname(pname);

	/* runtime.a -> runtime */
	if(p != nil)
		*p = '\0';

	if(ctxt->debugvlog > 1 && ctxt->bso)
		Bprint(ctxt->bso, "%5.2f addlib: %s %s pulls in %s\n", cputime(), obj, src, pname);

	addlibpath(ctxt, src, obj, pname, name);
}

/*
 * add library to library list.
 *	srcref: src file referring to package
 *	objref: object file referring to package
 *	file: object file, e.g., /home/rsc/go/pkg/container/vector.a
 *	pkg: package import path, e.g. container/vector
 */
void
addlibpath(Link *ctxt, char *srcref, char *objref, char *file, char *pkg)
{
	int i;
	Library *l;

	for(i=0; i<ctxt->libraryp; i++)
		if(strcmp(file, ctxt->library[i].file) == 0)
			return;

	if(ctxt->debugvlog > 1 && ctxt->bso)
		Bprint(ctxt->bso, "%5.2f addlibpath: srcref: %s objref: %s file: %s pkg: %s\n",
			cputime(), srcref, objref, file, pkg);

	if(ctxt->libraryp == ctxt->nlibrary){
		ctxt->nlibrary = 50 + 2*ctxt->libraryp;
		ctxt->library = erealloc(ctxt->library, sizeof ctxt->library[0] * ctxt->nlibrary);
	}

	l = &ctxt->library[ctxt->libraryp++];
	l->objref = estrdup(objref);
	l->srcref = estrdup(srcref);
	l->file = estrdup(file);
	l->pkg = estrdup(pkg);
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
}

uchar	fnuxi8[8];
uchar	fnuxi4[4];
uchar	inuxi1[1];
uchar	inuxi2[2];
uchar	inuxi4[4];
uchar	inuxi8[8];

#define	LOG	5
void
mkfwd(LSym *sym)
{
	Prog *p;
	int i;
	int32 dwn[LOG], cnt[LOG];
	Prog *lst[LOG];

	for(i=0; i<LOG; i++) {
		if(i == 0)
			cnt[i] = 1;
		else
			cnt[i] = LOG * cnt[i-1];
		dwn[i] = 1;
		lst[i] = nil;
	}
	i = 0;
	for(p = sym->text; p != nil && p->link != nil; p = p->link) {
		i--;
		if(i < 0)
			i = LOG-1;
		p->forwd = nil;
		dwn[i]--;
		if(dwn[i] <= 0) {
			dwn[i] = cnt[i];
			if(lst[i] != nil)
				lst[i]->forwd = p;
			lst[i] = p;
		}
	}
}

Prog*
copyp(Link *ctxt, Prog *q)
{
	Prog *p;

	p = ctxt->arch->prg();
	*p = *q;
	return p;
}

Prog*
appendp(Link *ctxt, Prog *q)
{
	Prog *p;

	p = ctxt->arch->prg();
	p->link = q->link;
	q->link = p;
	p->lineno = q->lineno;
	p->mode = q->mode;
	return p;
}

vlong
atolwhex(char *s)
{
	vlong n;
	int f;

	n = 0;
	f = 0;
	while(*s == ' ' || *s == '\t')
		s++;
	if(*s == '-' || *s == '+') {
		if(*s++ == '-')
			f = 1;
		while(*s == ' ' || *s == '\t')
			s++;
	}
	if(s[0]=='0' && s[1]){
		if(s[1]=='x' || s[1]=='X'){
			s += 2;
			for(;;){
				if(*s >= '0' && *s <= '9')
					n = n*16 + *s++ - '0';
				else if(*s >= 'a' && *s <= 'f')
					n = n*16 + *s++ - 'a' + 10;
				else if(*s >= 'A' && *s <= 'F')
					n = n*16 + *s++ - 'A' + 10;
				else
					break;
			}
		} else
			while(*s >= '0' && *s <= '7')
				n = n*8 + *s++ - '0';
	} else
		while(*s >= '0' && *s <= '9')
			n = n*10 + *s++ - '0';
	if(f)
		n = -n;
	return n;
}

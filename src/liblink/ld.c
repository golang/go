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
addlib(Link *ctxt, char *src, char *obj, char *pathname)
{
	char name[1024], pname[1024], *p;
	int i;

	if(strlen(pathname) >= sizeof name)
		sysfatal("addlib pathname too long");
	strcpy(name, pathname);
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

	if((!ctxt->windows && name[0] == '/') || (ctxt->windows && name[1] == ':'))
		snprint(pname, sizeof pname, "%s", name);
	else {
		// try dot, -L "libdir", and then goroot.
		for(i=0; i<ctxt->nlibdir; i++) {
			snprint(pname, sizeof pname, "%s/%s", ctxt->libdir[i], name);
			if(access(pname, AEXIST) >= 0)
				break;
		}
	}
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
nuxiinit(LinkArch *arch)
{
	int i, c;

	if(arch->endian != BigEndian && arch->endian != LittleEndian)
		sysfatal("unknown endian (%#x) for arch %s", arch->endian, arch->name);

	for(i=0; i<4; i++) {
		c = find1(arch->endian, i+1);
		if(arch->endian == LittleEndian) {
			if(i < 2)
				inuxi2[i] = c;
			if(i < 1)
				inuxi1[i] = c;
		} else {
			if(i >= 2)
				inuxi2[i-2] = c;
			if(i >= 3)
				inuxi1[i-3] = c;
		}
		inuxi4[i] = c;
		if(c == i) {
			inuxi8[i] = c;
			inuxi8[i+4] = c+4;
		} else {
			inuxi8[i] = c+4;
			inuxi8[i+4] = c;
		}
		fnuxi4[i] = c;
		if(c == i) {
			fnuxi8[i] = c;
			fnuxi8[i+4] = c+4;
		} else {
			fnuxi8[i] = c+4;
			fnuxi8[i+4] = c;
		}
	}
}

uchar	fnuxi8[8];
uchar	fnuxi4[4];
uchar	inuxi1[1];
uchar	inuxi2[2];
uchar	inuxi4[4];
uchar	inuxi8[8];

enum
{
	LOG = 5,
};
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

	USED(ctxt);
	p = emallocz(sizeof(Prog));
	*p = *q;
	return p;
}

Prog*
appendp(Link *ctxt, Prog *q)
{
	Prog *p;

	USED(ctxt);
	p = emallocz(sizeof(Prog));
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

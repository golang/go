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

static int
yy_isalpha(int c)
{
	return c >= 0 && c <= 0xFF && isalpha(c);
}

Link*
linknew(LinkArch *arch)
{
	Link *ctxt;
	char *p;
	char buf[1024];

	nuxiinit();
	
	ctxt = emallocz(sizeof *ctxt);
	ctxt->arch = arch;
	ctxt->version = HistVersion;

	// TODO: Make caller pass in ctxt->arch,
	// so that for example 6g only has the linkamd64 code.
	p = getgoarch();
	if(strncmp(p, arch->name, strlen(arch->name)) != 0)
		sysfatal("invalid goarch %s (want %s or derivative)", p, arch->name);
	
	if(getwd(buf, sizeof buf) == 0)
		strcpy(buf, "/???");
	if(yy_isalpha(buf[0]) && buf[1] == ':') {
		// On Windows.
		ctxt->windows = 1;

		// Canonicalize path by converting \ to / (Windows accepts both).
		for(p=buf; *p; p++)
			if(*p == '\\')
				*p = '/';
	}
	ctxt->pathname = strdup(buf);

	return ctxt;
}

LSym*
linknewsym(Link *ctxt, char *symb, int v)
{
	LSym *s;
	int l;

	l = strlen(symb) + 1;
	s = malloc(sizeof(*s));
	memset(s, 0, sizeof(*s));

	s->dynid = -1;
	s->plt = -1;
	s->got = -1;
	s->name = malloc(l + 1);
	memmove(s->name, symb, l);
	s->name[l] = '\0';

	s->type = 0;
	s->version = v;
	s->value = 0;
	s->sig = 0;
	s->size = 0;
	ctxt->nsymbol++;

	s->allsym = ctxt->allsym;
	ctxt->allsym = s;

	return s;
}

static LSym*
_lookup(Link *ctxt, char *symb, int v, int creat)
{
	LSym *s;
	char *p;
	uint32 h;
	int c;

	h = v;
	for(p=symb; c = *p; p++)
		h = h+h+h + c;
	h &= 0xffffff;
	h %= LINKHASH;
	for(s = ctxt->hash[h]; s != nil; s = s->hash)
		if(strcmp(s->name, symb) == 0)
			return s;
	if(!creat)
		return nil;

	s = linknewsym(ctxt, symb, v);
	s->extname = s->name;
	s->hash = ctxt->hash[h];
	ctxt->hash[h] = s;

	return s;
}

LSym*
linklookup(Link *ctxt, char *name, int v)
{
	return _lookup(ctxt, name, v, 1);
}

// read-only lookup
LSym*
linkrlookup(Link *ctxt, char *name, int v)
{
	return _lookup(ctxt, name, v, 0);
}

int
linksymfmt(Fmt *f)
{
	LSym *s;
	
	s = va_arg(f->args, LSym*);
	if(s == nil)
		return fmtstrcpy(f, "<nil>");
	
	return fmtstrcpy(f, s->name);
}

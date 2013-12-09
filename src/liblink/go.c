// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go-specific code shared across loaders (5l, 6l, 8l).

#include <u.h>
#include <libc.h>
#include <bio.h>
#include <link.h>

// replace all "". with pkg.
char*
expandpkg(char *t0, char *pkg)
{
	int n;
	char *p;
	char *w, *w0, *t;

	n = 0;
	for(p=t0; (p=strstr(p, "\"\".")) != nil; p+=3)
		n++;

	if(n == 0)
		return estrdup(t0);

	w0 = emallocz(strlen(t0) + strlen(pkg)*n);
	w = w0;
	for(p=t=t0; (p=strstr(p, "\"\".")) != nil; p=t) {
		memmove(w, t, p - t);
		w += p-t;
		strcpy(w, pkg);
		w += strlen(pkg);
		t = p+2;
	}
	strcpy(w, t);
	return w0;
}

void*
emallocz(long n)
{
	void *p;

	p = malloc(n);
	if(p == nil)
		sysfatal("out of memory");
	memset(p, 0, n);
	return p;
}

char*
estrdup(char *p)
{
	p = strdup(p);
	if(p == nil)
		sysfatal("out of memory");
	return p;
}

void*
erealloc(void *p, long n)
{
	p = realloc(p, n);
	if(p == nil)
		sysfatal("out of memory");
	return p;
}

void
double2ieee(uint64 *ieee, float64 f)
{
	memmove(ieee, &f, 8);
}

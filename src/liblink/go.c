// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go-specific code shared across loaders (5l, 6l, 8l).

#include <u.h>
#include <libc.h>
#include <bio.h>
#include <link.h>

int framepointer_enabled;
int fieldtrack_enabled;

// Toolchain experiments.
// These are controlled by the GOEXPERIMENT environment
// variable recorded when the toolchain is built.
// This list is also known to cmd/gc.
static struct {
	char *name;
	int *val;
} exper[] = {
	{"fieldtrack", &fieldtrack_enabled},
	{"basepointer", &framepointer_enabled}, 
};

static void
addexp(char *s)
{
	int i;

	for(i=0; i < nelem(exper); i++ ) {
		if(strcmp(exper[i].name, s) == 0) {
			if(exper[i].val != nil)
				*exper[i].val = 1;
			return;
		}
	}
	
	print("unknown experiment %s\n", s);
	exits("unknown experiment");
}

void
linksetexp(void)
{
	char *f[20];
	int i, nf;

	// cmd/dist #defines GOEXPERIMENT for us.
	nf = getfields(GOEXPERIMENT, f, nelem(f), 1, ",");
	for(i=0; i<nf; i++)
		addexp(f[i]);
}

char*
expstring(void)
{
	int i;
	static char buf[512];

	strcpy(buf, "X");
	for(i=0; i<nelem(exper); i++)
		if(*exper[i].val)
			seprint(buf+strlen(buf), buf+sizeof buf, ",%s", exper[i].name);
	if(strlen(buf) == 1)
		strcpy(buf, "X,none");
	buf[1] = ':';
	return buf;
}

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

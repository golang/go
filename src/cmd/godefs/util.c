// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "a.h"

void*
emalloc(int n)
{
	void *p;

	p = malloc(n);
	if(p == nil)
		sysfatal("out of memory");
	memset(p, 0, n);
	return p;
}

char*
estrdup(char *s)
{
	s = strdup(s);
	if(s == nil)
		sysfatal("out of memory");
	return s;
}

void*
erealloc(void *v, int n)
{
	v = realloc(v, n);
	if(v == nil)
		sysfatal("out of memory");
	return v;
}


// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

static	int32	empty		= 0;
string	emptystring	= (string)&empty;

int32
findnull(byte *s)
{
	int32 l;

	for(l=0; s[l]!=0; l++)
		;
	return l;
}

string
gostring(byte *str)
{
	int32 l;
	string s;

	l = findnull(str);
	s = mal(sizeof(s->len)+l+1);
	s->len = l;
	mcpy(s->str, str, l+1);
	return s;
}

void
sys·catstring(string s1, string s2, string s3)
{
	uint32 l;

	if(s1 == nil || s1->len == 0) {
		s3 = s2;
		goto out;
	}
	if(s2 == nil || s2->len == 0) {
		s3 = s1;
		goto out;
	}

	l = s1->len + s2->len;

	s3 = mal(sizeof(s3->len)+l);
	s3->len = l;
	mcpy(s3->str, s1->str, s1->len);
	mcpy(s3->str+s1->len, s2->str, s2->len);

out:
	FLUSH(&s3);
}

static void
prbounds(int8* s, int32 a, int32 b, int32 c)
{
	prints(s);
	prints(" ");
	sys·printint(a);
	prints("<");
	sys·printint(b);
	prints(">");
	sys·printint(c);
	prints("\n");
	throw("string bounds");
}

uint32
cmpstring(string s1, string s2)
{
	uint32 i, l;
	byte c1, c2;

	if(s1 == nil)
		s1 = emptystring;
	if(s2 == nil)
		s2 = emptystring;

	l = s1->len;
	if(s2->len < l)
		l = s2->len;
	for(i=0; i<l; i++) {
		c1 = s1->str[i];
		c2 = s2->str[i];
		if(c1 < c2)
			return -1;
		if(c1 > c2)
			return +1;
	}
	if(s1->len < s2->len)
		return -1;
	if(s1->len > s2->len)
		return +1;
	return 0;
}

void
sys·cmpstring(string s1, string s2, int32 v)
{
	v = cmpstring(s1, s2);
	FLUSH(&v);
}

int32
strcmp(byte *s1, byte *s2)
{
	uint32 i;
	byte c1, c2;

	for(i=0;; i++) {
		c1 = s1[i];
		c2 = s2[i];
		if(c1 < c2)
			return -1;
		if(c1 > c2)
			return +1;
		if(c1 == 0)
			return 0;
	}
}

void
sys·slicestring(string si, int32 lindex, int32 hindex, string so)
{
	int32 l;

	if(si == nil)
		si = emptystring;

	if(lindex < 0 || lindex > si->len ||
	   hindex < lindex || hindex > si->len) {
		sys·printpc(&si);
		prints(" ");
		prbounds("slice", lindex, si->len, hindex);
	}

	l = hindex-lindex;
	so = mal(sizeof(so->len)+l);
	so->len = l;
	mcpy(so->str, si->str+lindex, l);
	FLUSH(&so);
}

void
sys·indexstring(string s, int32 i, byte b)
{
	if(s == nil)
		s = emptystring;

	if(i < 0 || i >= s->len) {
		sys·printpc(&s);
		prints(" ");
		prbounds("index", 0, i, s->len);
	}

	b = s->str[i];
	FLUSH(&b);
}

void
sys·intstring(int64 v, string s)
{
	s = mal(sizeof(s->len)+8);
	s->len = runetochar(s->str, v);
	FLUSH(&s);
}

void
sys·byteastring(byte *a, int32 l, string s)
{
	s = mal(sizeof(s->len)+l);
	s->len = l;
	mcpy(s->str, a, l);
	FLUSH(&s);
}

void
sys·arraystring(Array *b, string s)
{
	s = mal(sizeof(s->len)+b->nel);
	s->len = b->nel;
	mcpy(s->str, b->array, s->len);
	FLUSH(&s);
}

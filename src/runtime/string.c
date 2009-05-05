// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

String	emptystring;

int32
findnull(byte *s)
{
	int32 l;

	if(s == nil)
		return 0;
	for(l=0; s[l]!=0; l++)
		;
	return l;
}

int32 maxstring;

String
gostringsize(int32 l)
{
	String s;

	if(l == 0)
		return emptystring;
	s.str = mal(l);
	s.len = l;
	if(l > maxstring)
		maxstring = l;
	return s;
}

String
gostring(byte *str)
{
	int32 l;
	String s;

	l = findnull(str);
	s = gostringsize(l);
	mcpy(s.str, str, l);
	return s;
}

void
sys·catstring(String s1, String s2, String s3)
{
	if(s1.len == 0) {
		s3 = s2;
		goto out;
	}
	if(s2.len == 0) {
		s3 = s1;
		goto out;
	}

	s3 = gostringsize(s1.len + s2.len);
	mcpy(s3.str, s1.str, s1.len);
	mcpy(s3.str+s1.len, s2.str, s2.len);

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
cmpstring(String s1, String s2)
{
	uint32 i, l;
	byte c1, c2;

	l = s1.len;
	if(s2.len < l)
		l = s2.len;
	for(i=0; i<l; i++) {
		c1 = s1.str[i];
		c2 = s2.str[i];
		if(c1 < c2)
			return -1;
		if(c1 > c2)
			return +1;
	}
	if(s1.len < s2.len)
		return -1;
	if(s1.len > s2.len)
		return +1;
	return 0;
}

void
sys·cmpstring(String s1, String s2, int32 v)
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
sys·slicestring(String si, int32 lindex, int32 hindex, String so)
{
	int32 l;

	if(lindex < 0 || lindex > si.len ||
	   hindex < lindex || hindex > si.len) {
		sys·printpc(&si);
		prints(" ");
		prbounds("slice", lindex, si.len, hindex);
	}

	l = hindex-lindex;
	so.str = si.str + lindex;
	so.len = l;

//	alternate to create a new string
//	so = gostringsize(l);
//	mcpy(so.str, si.str+lindex, l);

	FLUSH(&so);
}

void
sys·indexstring(String s, int32 i, byte b)
{
	if(i < 0 || i >= s.len) {
		sys·printpc(&s);
		prints(" ");
		prbounds("index", 0, i, s.len);
	}

	b = s.str[i];
	FLUSH(&b);
}

void
sys·intstring(int64 v, String s)
{
	s = gostringsize(8);
	s.len = runetochar(s.str, v);
	FLUSH(&s);
}

void
sys·arraystring(Array b, String s)
{
	s = gostringsize(b.nel);
	mcpy(s.str, b.array, s.len);
	FLUSH(&s);
}

enum
{
	Runeself	= 0x80,
};

// func	stringiter(string, int) (retk int);
void
sys·stringiter(String s, int32 k, int32 retk)
{
	int32 l;

	if(k >= s.len) {
		// retk=0 is end of iteration
		retk = 0;
		goto out;
	}

	l = s.str[k];
	if(l < Runeself) {
		retk = k+1;
		goto out;
	}

	// multi-char rune
	retk = k + charntorune(&l, s.str+k, s.len-k);

out:
	FLUSH(&retk);
}

// func	stringiter2(string, int) (retk int, retv any);
void
sys·stringiter2(String s, int32 k, int32 retk, int32 retv)
{
	int32 l;

	if(k >= s.len) {
		// retk=0 is end of iteration
		retk = 0;
		retv = 0;
		goto out;
	}

	retv = s.str[k];
	if(retv < Runeself) {
		retk = k+1;
		goto out;
	}

	// multi-char rune
	retk = k + charntorune(&retv, s.str+k, s.len-k);

out:
	FLUSH(&retk);
	FLUSH(&retv);
}

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"
#include "race.h"
#include "../../cmd/ld/textflag.h"

String	runtime·emptystring;

#pragma textflag NOSPLIT
intgo
runtime·findnull(byte *s)
{
	intgo l;

	if(s == nil)
		return 0;
	for(l=0; s[l]!=0; l++)
		;
	return l;
}

intgo
runtime·findnullw(uint16 *s)
{
	intgo l;

	if(s == nil)
		return 0;
	for(l=0; s[l]!=0; l++)
		;
	return l;
}

uintptr runtime·maxstring = 256; // a hint for print

static String
gostringsize(intgo l)
{
	String s;
	uintptr ms;

	if(l == 0)
		return runtime·emptystring;
	s.str = runtime·mallocgc(l, 0, FlagNoScan|FlagNoZero);
	s.len = l;
	for(;;) {
		ms = runtime·maxstring;
		if((uintptr)l <= ms || runtime·casp((void**)&runtime·maxstring, (void*)ms, (void*)l))
			break;
	}
	return s;
}

String
runtime·gostring(byte *str)
{
	intgo l;
	String s;

	l = runtime·findnull(str);
	s = gostringsize(l);
	runtime·memmove(s.str, str, l);
	return s;
}

String
runtime·gostringn(byte *str, intgo l)
{
	String s;

	s = gostringsize(l);
	runtime·memmove(s.str, str, l);
	return s;
}

// used by cmd/cgo
Slice
runtime·gobytes(byte *p, intgo n)
{
	Slice sl;

	sl.array = runtime·mallocgc(n, 0, FlagNoScan|FlagNoZero);
	sl.len = n;
	sl.cap = n;
	runtime·memmove(sl.array, p, n);
	return sl;
}

String
runtime·gostringnocopy(byte *str)
{
	String s;
	
	s.str = str;
	s.len = runtime·findnull(str);
	return s;
}

String
runtime·gostringw(uint16 *str)
{
	intgo n1, n2, i;
	byte buf[8];
	String s;

	n1 = 0;
	for(i=0; str[i]; i++)
		n1 += runtime·runetochar(buf, str[i]);
	s = gostringsize(n1+4);
	n2 = 0;
	for(i=0; str[i]; i++) {
		// check for race
		if(n2 >= n1)
			break;
		n2 += runtime·runetochar(s.str+n2, str[i]);
	}
	s.len = n2;
	s.str[s.len] = 0;
	return s;
}

String
runtime·catstring(String s1, String s2)
{
	String s3;

	if(s1.len == 0)
		return s2;
	if(s2.len == 0)
		return s1;

	s3 = gostringsize(s1.len + s2.len);
	runtime·memmove(s3.str, s1.str, s1.len);
	runtime·memmove(s3.str+s1.len, s2.str, s2.len);
	return s3;
}

int32
runtime·strcmp(byte *s1, byte *s2)
{
	uintptr i;
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

int32
runtime·strncmp(byte *s1, byte *s2, uintptr n)
{
	uintptr i;
	byte c1, c2;

	for(i=0; i<n; i++) {
		c1 = s1[i];
		c2 = s2[i];
		if(c1 < c2)
			return -1;
		if(c1 > c2)
			return +1;
		if(c1 == 0)
			break;
	}
	return 0;
}

byte*
runtime·strstr(byte *s1, byte *s2)
{
	byte *sp1, *sp2;

	if(*s2 == 0)
		return s1;
	for(; *s1; s1++) {
		if(*s1 != *s2)
			continue;
		sp1 = s1;
		sp2 = s2;
		for(;;) {
			if(*sp2 == 0)
				return s1;
			if(*sp1++ != *sp2++)
				break;
		}
	}
	return nil;
}

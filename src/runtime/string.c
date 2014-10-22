// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "malloc.h"
#include "race.h"
#include "textflag.h"

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

#pragma textflag NOSPLIT
String
runtime·gostringnocopy(byte *str)
{
	String s;
	uintptr ms;
	
	s.str = str;
	s.len = runtime·findnull(str);
	while(true) {
		ms = runtime·maxstring;
		if(s.len <= ms || runtime·casp((void**)&runtime·maxstring, (void*)ms, (void*)s.len))
			return s;
	}
}

// TODO: move this elsewhere
enum
{
	Bit1	= 7,
	Bitx	= 6,
	Bit2	= 5,
	Bit3	= 4,
	Bit4	= 3,
	Bit5	= 2,

	Tx	= ((1<<(Bitx+1))-1) ^ 0xFF,	/* 1000 0000 */
	T2	= ((1<<(Bit2+1))-1) ^ 0xFF,	/* 1100 0000 */
	T3	= ((1<<(Bit3+1))-1) ^ 0xFF,	/* 1110 0000 */
	T4	= ((1<<(Bit4+1))-1) ^ 0xFF,	/* 1111 0000 */

	Rune1	= (1<<(Bit1+0*Bitx))-1,		/* 0000 0000 0111 1111 */
	Rune2	= (1<<(Bit2+1*Bitx))-1,		/* 0000 0111 1111 1111 */
	Rune3	= (1<<(Bit3+2*Bitx))-1,		/* 1111 1111 1111 1111 */

	Maskx	= (1<<Bitx)-1,			/* 0011 1111 */

	Runeerror	= 0xFFFD,

	SurrogateMin = 0xD800,
	SurrogateMax = 0xDFFF,

	Runemax	= 0x10FFFF,	/* maximum rune value */
};

static int32
runetochar(byte *str, int32 rune)  /* note: in original, arg2 was pointer */
{
	/* Runes are signed, so convert to unsigned for range check. */
	uint32 c;

	/*
	 * one character sequence
	 *	00000-0007F => 00-7F
	 */
	c = rune;
	if(c <= Rune1) {
		str[0] = c;
		return 1;
	}

	/*
	 * two character sequence
	 *	0080-07FF => T2 Tx
	 */
	if(c <= Rune2) {
		str[0] = T2 | (c >> 1*Bitx);
		str[1] = Tx | (c & Maskx);
		return 2;
	}

	/*
	 * If the Rune is out of range or a surrogate half, convert it to the error rune.
	 * Do this test here because the error rune encodes to three bytes.
	 * Doing it earlier would duplicate work, since an out of range
	 * Rune wouldn't have fit in one or two bytes.
	 */
	if (c > Runemax)
		c = Runeerror;
	if (SurrogateMin <= c && c <= SurrogateMax)
		c = Runeerror;

	/*
	 * three character sequence
	 *	0800-FFFF => T3 Tx Tx
	 */
	if (c <= Rune3) {
		str[0] = T3 |  (c >> 2*Bitx);
		str[1] = Tx | ((c >> 1*Bitx) & Maskx);
		str[2] = Tx |  (c & Maskx);
		return 3;
	}

	/*
	 * four character sequence (21-bit value)
	 *     10000-1FFFFF => T4 Tx Tx Tx
	 */
	str[0] = T4 | (c >> 3*Bitx);
	str[1] = Tx | ((c >> 2*Bitx) & Maskx);
	str[2] = Tx | ((c >> 1*Bitx) & Maskx);
	str[3] = Tx | (c & Maskx);
	return 4;
}

String runtime·gostringsize(intgo);

String
runtime·gostringw(uint16 *str)
{
	intgo n1, n2, i;
	byte buf[8];
	String s;

	n1 = 0;
	for(i=0; str[i]; i++)
		n1 += runetochar(buf, str[i]);
	s = runtime·gostringsize(n1+4);
	n2 = 0;
	for(i=0; str[i]; i++) {
		// check for race
		if(n2 >= n1)
			break;
		n2 += runetochar(s.str+n2, str[i]);
	}
	s.len = n2;
	s.str[s.len] = 0;
	return s;
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

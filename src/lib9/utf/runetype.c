/*
 * The authors of this software are Rob Pike and Ken Thompson.
 *              Copyright (c) 2002 by Lucent Technologies.
 * Permission to use, copy, modify, and distribute this software for any
 * purpose without fee is hereby granted, provided that this entire notice
 * is included in all copies of any software which is or includes a copy
 * or modification of this software and in all copies of the supporting
 * documentation for such software.
 * THIS SOFTWARE IS BEING PROVIDED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED
 * WARRANTY.  IN PARTICULAR, NEITHER THE AUTHORS NOR LUCENT TECHNOLOGIES MAKE ANY
 * REPRESENTATION OR WARRANTY OF ANY KIND CONCERNING THE MERCHANTABILITY
 * OF THIS SOFTWARE OR ITS FITNESS FOR ANY PARTICULAR PURPOSE.
 */
#include "utf.h"
#include "utfdef.h"

static
Rune*
rbsearch(Rune c, Rune *t, int n, int ne)
{
	Rune *p;
	int m;

	while(n > 1) {
		m = n >> 1;
		p = t + m*ne;
		if(c >= p[0]) {
			t = p;
			n = n-m;
		} else
			n = m;
	}
	if(n && c >= t[0])
		return t;
	return 0;
}

/*
 * The "ideographic" property is hard to extract from UnicodeData.txt,
 * so it is hard coded here.
 *
 * It is defined in the Unicode PropList.txt file, for example
 * PropList-3.0.0.txt.  Unlike the UnicodeData.txt file, the format of
 * PropList changes between versions.  This property appears relatively static;
 * it is the same in version 4.0.1, except that version defines some >16 bit
 * chars as ideographic as well: 20000..2a6d6, and 2f800..2Fa1d.
 */
static Rune __isideographicr[] = {
	0x3006, 0x3007,		/* 0x3006 added in 2.0.14 */
	0x3021, 0x3029,
	0x3038, 0x303a,		/* added in 3.0.0 */
	0x3400, 0x4db5,		/* added in 3.0.0 */

	/* consecutive */
	0x4e00, 0x9fa5,
	0x9fa6, 0x9fbb,		/* added in 4.1.0 */
	0x9fbc, 0x9fc3,		/* added in 5.1.0 */
	0x9fc4, 0x9fcb,		/* added in 5.2.0 */

	0xf900, 0xfa2d,

	/* consecutive */
	0xfa30, 0xfa6a,		/* added in 5.1.0 */
	0xfa6b, 0xfa6d,		/* added in 5.2.0 */

	0xfa70, 0xfad9,		/* added in 4.1.0 */
	0x20000, 0x2a6d6,	/* added in 3.1.0 */
	0x2a700, 0x2b734,	/* added in 5.2.0 */
	0x2f800, 0x2fa1d,	/* added in 3.1.0 */
};

int
isideographicrune(Rune c)
{
	Rune *p;

	p = rbsearch(c, __isideographicr, nelem(__isideographicr)/2, 2);
	if(p && c >= p[0] && c <= p[1])
		return 1;
	return 0;
}

#include "runetypebody-5.2.0.c"

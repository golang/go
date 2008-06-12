/*
 * The authors of this software are Rob Pike and Ken Thompson.
 *              Copyright (c) 2002 by Lucent Technologies.
 * Permission to use, copy, modify, and distribute this software for any
 * purpose without fee is hereby granted, provided that this entire notice
 * is included in all copies of any software which is or includes a copy
 * or modification of this software and in all copies of the supporting
 * documentation for such software.
 * THIS SOFTWARE IS BEING PROVIDED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED
 * WARRANTY.  IN PARTICULAR, NEITHER THE AUTHORS NOR LUCENT TECHNOLOGIES MAKE
 * ANY REPRESENTATION OR WARRANTY OF ANY KIND CONCERNING THE MERCHANTABILITY
 * OF THIS SOFTWARE OR ITS FITNESS FOR ANY PARTICULAR PURPOSE.
 */
#include <stdarg.h>
#include <string.h>
#include "plan9.h"
#include "utf.h"

char*
utfrrune(char *s, long c)
{
	long c1;
	Rune r;
	char *s1;

	if(c < Runesync)		/* not part of utf sequence */
		return strrchr(s, c);

	s1 = 0;
	for(;;) {
		c1 = *(uchar*)s;
		if(c1 < Runeself) {	/* one byte rune */
			if(c1 == 0)
				return s1;
			if(c1 == c)
				s1 = s;
			s++;
			continue;
		}
		c1 = chartorune(&r, s);
		if(r == c)
			s1 = s;
		s += c1;
	}
}

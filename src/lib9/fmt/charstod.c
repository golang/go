/*
 * The authors of this software are Rob Pike and Ken Thompson,
 * with contributions from Mike Burrows and Sean Dorward.
 *
 *     Copyright (c) 2002-2006 by Lucent Technologies.
 *     Portions Copyright (c) 2004 Google Inc.
 * 
 * Permission to use, copy, modify, and distribute this software for any
 * purpose without fee is hereby granted, provided that this entire notice
 * is included in all copies of any software which is or includes a copy
 * or modification of this software and in all copies of the supporting
 * documentation for such software.
 * THIS SOFTWARE IS BEING PROVIDED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED
 * WARRANTY.  IN PARTICULAR, NEITHER THE AUTHORS NOR LUCENT TECHNOLOGIES 
 * NOR GOOGLE INC MAKE ANY REPRESENTATION OR WARRANTY OF ANY KIND CONCERNING 
 * THE MERCHANTABILITY OF THIS SOFTWARE OR ITS FITNESS FOR ANY PARTICULAR PURPOSE.
 */

#include <u.h>
#include <libc.h>
#include "fmtdef.h"

/*
 * Reads a floating-point number by interpreting successive characters
 * returned by (*f)(vp).  The last call it makes to f terminates the
 * scan, so is not a character in the number.  It may therefore be
 * necessary to back up the input stream up one byte after calling charstod.
 */

double
fmtcharstod(int(*f)(void*), void *vp)
{
	double num, dem;
	int neg, eneg, dig, exp, c;

	num = 0;
	neg = 0;
	dig = 0;
	exp = 0;
	eneg = 0;

	c = (*f)(vp);
	while(c == ' ' || c == '\t')
		c = (*f)(vp);
	if(c == '-' || c == '+'){
		if(c == '-')
			neg = 1;
		c = (*f)(vp);
	}
	while(c >= '0' && c <= '9'){
		num = num*10 + c-'0';
		c = (*f)(vp);
	}
	if(c == '.')
		c = (*f)(vp);
	while(c >= '0' && c <= '9'){
		num = num*10 + c-'0';
		dig++;
		c = (*f)(vp);
	}
	if(c == 'e' || c == 'E'){
		c = (*f)(vp);
		if(c == '-' || c == '+'){
			if(c == '-'){
				dig = -dig;
				eneg = 1;
			}
			c = (*f)(vp);
		}
		while(c >= '0' && c <= '9'){
			exp = exp*10 + c-'0';
			c = (*f)(vp);
		}
	}
	exp -= dig;
	if(exp < 0){
		exp = -exp;
		eneg = !eneg;
	}
	dem = __fmtpow10(exp);
	if(eneg)
		num /= dem;
	else
		num *= dem;
	if(neg)
		return -num;
	return num;
}

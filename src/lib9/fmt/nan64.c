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

/*
 * 64-bit IEEE not-a-number routines.
 * This is big/little-endian portable assuming that
 * the 64-bit doubles and 64-bit integers have the
 * same byte ordering.
 */

#include <u.h>
#include <libc.h>
#include "fmtdef.h"

static uvlong uvnan    = ((uvlong)0x7FF00000<<32)|0x00000001;
static uvlong uvinf    = ((uvlong)0x7FF00000<<32)|0x00000000;
static uvlong uvneginf = ((uvlong)0xFFF00000<<32)|0x00000000;

/* gcc sees through the obvious casts. */
static uvlong
d2u(double d)
{
	union {
		uvlong v;
		double d;
	} u;
	assert(sizeof(u.d) == sizeof(u.v));
	u.d = d;
	return u.v;
}

static double
u2d(uvlong v)
{
	union {
		uvlong v;
		double d;
	} u;
	assert(sizeof(u.d) == sizeof(u.v));
	u.v = v;
	return u.d;
}

double
__NaN(void)
{
	return u2d(uvnan);
}

int
__isNaN(double d)
{
	uvlong x;

	x = d2u(d);
	/* IEEE 754: exponent bits 0x7FF and non-zero mantissa */
	return (x&uvinf) == uvinf && (x&~uvneginf) != 0;
}

double
__Inf(int sign)
{
	return u2d(sign < 0 ? uvneginf : uvinf);
}

int
__isInf(double d, int sign)
{
	uvlong x;

	x = d2u(d);
	if(sign == 0)
		return x==uvinf || x==uvneginf;
	else if(sign > 0)
		return x==uvinf;
	else
		return x==uvneginf;
}

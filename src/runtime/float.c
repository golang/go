// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

static	uint64	uvnan		= 0x7FF0000000000001ULL;
static	uint64	uvinf		= 0x7FF0000000000000ULL;
static	uint64	uvneginf	= 0xFFF0000000000000ULL;

uint32
float32tobits(float32 f)
{
	// The obvious cast-and-pointer code is technically
	// not valid, and gcc miscompiles it.  Use a union instead.
	union {
		float32 f;
		uint32 i;
	} u;
	u.f = f;
	return u.i;
}

uint64
float64tobits(float64 f)
{
	// The obvious cast-and-pointer code is technically
	// not valid, and gcc miscompiles it.  Use a union instead.
	union {
		float64 f;
		uint64 i;
	} u;
	u.f = f;
	return u.i;
}

float64
float64frombits(uint64 i)
{
	// The obvious cast-and-pointer code is technically
	// not valid, and gcc miscompiles it.  Use a union instead.
	union {
		float64 f;
		uint64 i;
	} u;
	u.i = i;
	return u.f;
}

float32
float32frombits(uint32 i)
{
	// The obvious cast-and-pointer code is technically
	// not valid, and gcc miscompiles it.  Use a union instead.
	union {
		float32 f;
		uint32 i;
	} u;
	u.i = i;
	return u.f;
}

bool
isInf(float64 f, int32 sign)
{
	uint64 x;

	x = float64tobits(f);
	if(sign == 0)
		return x == uvinf || x == uvneginf;
	if(sign > 0)
		return x == uvinf;
	return x == uvneginf;
}

float64
NaN(void)
{
	return float64frombits(uvnan);
}

bool
isNaN(float64 f)
{
	uint64 x;

	x = float64tobits(f);
	return ((uint32)(x>>52) & 0x7FF) == 0x7FF && !isInf(f, 0);
}

float64
Inf(int32 sign)
{
	if(sign >= 0)
		return float64frombits(uvinf);
	else
		return float64frombits(uvneginf);
}

enum
{
	MASK	= 0x7ffL,
	SHIFT	= 64-11-1,
	BIAS	= 1022L,
};

float64
frexp(float64 d, int32 *ep)
{
	uint64 x;

	if(d == 0) {
		*ep = 0;
		return 0;
	}
	x = float64tobits(d);
	*ep = (int32)((x >> SHIFT) & MASK) - BIAS;
	x &= ~((uint64)MASK << SHIFT);
	x |= (uint64)BIAS << SHIFT;
	return float64frombits(x);
}

float64
ldexp(float64 d, int32 e)
{
	uint64 x;

	if(d == 0)
		return 0;
	x = float64tobits(d);
	e += (int32)(x >> SHIFT) & MASK;
	if(e <= 0)
		return 0;	/* underflow */
	if(e >= MASK){		/* overflow */
		if(d < 0)
			return Inf(-1);
		return Inf(1);
	}
	x &= ~((uint64)MASK << SHIFT);
	x |= (uint64)e << SHIFT;
	return float64frombits(x);
}

float64
modf(float64 d, float64 *ip)
{
	float64 dd;
	uint64 x;
	int32 e;

	if(d < 1) {
		if(d < 0) {
			d = modf(-d, ip);
			*ip = -*ip;
			return -d;
		}
		*ip = 0;
		return d;
	}

	x = float64tobits(d);
	e = (int32)((x >> SHIFT) & MASK) - BIAS;

	/*
	 * Keep the top 11+e bits; clear the rest.
	 */
	if(e <= 64-11)
		x &= ~(((uint64)1 << (64LL-11LL-e))-1);
	dd = float64frombits(x);
	*ip = dd;
	return d - dd;
}


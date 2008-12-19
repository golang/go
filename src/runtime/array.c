// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

static	int32	debug	= 0;

// newarray(nel int, cap int, width int) (ary []any);
void
sys·newarray(uint32 nel, uint32 cap, uint32 width, Array ret)
{
	uint64 size;

	if(cap < nel)
		cap = nel;
	size = cap*width;

	ret.nel = nel;
	ret.cap = cap;
	ret.array = mal(size);

	FLUSH(&ret);

	if(debug) {
		prints("newarray: nel=");
		sys·printint(nel);
		prints("; cap=");
		sys·printint(cap);
		prints("; width=");
		sys·printint(width);
		prints("; ret=");
		sys·printarray(&ret);
		prints("\n");
	}
}

static void
throwslice(uint32 lb, uint32 hb, uint32 n)
{
	prints("slice[");
	sys·printint(lb);
	prints(":");
	sys·printint(hb);
	prints("] of [");
	sys·printint(n);
	prints("] array\n");
	throw("array slice");
}

// arraysliced(old []any, lb int, hb int, width int) (ary []any);
void
sys·arraysliced(Array old, uint32 lb, uint32 hb, uint32 width, Array ret)
{

	if(hb > old.cap || lb > hb) {
		if(debug) {
			prints("sys·arraysliced: old=");
			sys·printarray(&old);
			prints("; lb=");
			sys·printint(lb);
			prints("; hb=");
			sys·printint(hb);
			prints("; width=");
			sys·printint(width);
			prints("\n");

			prints("oldarray: nel=");
			sys·printint(old.nel);
			prints("; cap=");
			sys·printint(old.cap);
			prints("\n");
		}
		throwslice(lb, hb, old.cap);
	}

	// new array is inside old array
	ret.nel = hb-lb;
	ret.cap = old.cap - lb;
	ret.array = old.array + lb*width;

	FLUSH(&ret);

	if(debug) {
		prints("sys·arraysliced: old=");
		sys·printarray(&old);
		prints("; lb=");
		sys·printint(lb);
		prints("; hb=");
		sys·printint(hb);
		prints("; width=");
		sys·printint(width);
		prints("; ret=");
		sys·printarray(&ret);
		prints("\n");
	}
}

// arrayslices(old *any, nel int, lb int, hb int, width int) (ary []any);
void
sys·arrayslices(byte* old, uint32 nel, uint32 lb, uint32 hb, uint32 width, Array ret)
{

	if(hb > nel || lb > hb) {
		if(debug) {
			prints("sys·arrayslices: old=");
			sys·printpointer(old);
			prints("; nel=");
			sys·printint(nel);
			prints("; lb=");
			sys·printint(lb);
			prints("; hb=");
			sys·printint(hb);
			prints("; width=");
			sys·printint(width);
			prints("\n");
		}
		throwslice(lb, hb, nel);
	}

	// new array is inside old array
	ret.nel = hb-lb;
	ret.cap = nel-lb;
	ret.array = old + lb*width;

	FLUSH(&ret);

	if(debug) {
		prints("sys·arrayslices: old=");
		sys·printpointer(old);
		prints("; nel=");
		sys·printint(nel);
		prints("; lb=");
		sys·printint(lb);
		prints("; hb=");
		sys·printint(hb);
		prints("; width=");
		sys·printint(width);
		prints("; ret=");
		sys·printarray(&ret);
		prints("\n");
	}
}

// arrays2d(old *any, nel int) (ary []any)
void
sys·arrays2d(byte* old, uint32 nel, Array ret)
{

	// new dope to old array
	ret.nel = nel;
	ret.cap = nel;
	ret.array = old;

	FLUSH(&ret);

	if(debug) {
		prints("sys·arrays2d: old=");
		sys·printpointer(old);
		prints("; ret=");
		sys·printarray(&ret);
		prints("\n");
	}
}

void
sys·printarray(Array *a)
{
	prints("[");
	sys·printint(a->nel);
	prints("/");
	sys·printint(a->cap);
	prints("]");
	sys·printpointer(a->array);
}

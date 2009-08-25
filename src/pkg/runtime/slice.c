// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

static	int32	debug	= 0;

// makeslice(nel int, cap int, width int) (ary []any);
void
sys·makeslice(uint32 nel, uint32 cap, uint32 width, Slice ret)
{
	uint64 size;

	if(cap < nel)
		cap = nel;
	size = cap*width;

	ret.len = nel;
	ret.cap = cap;
	ret.array = mal(size);

	FLUSH(&ret);

	if(debug) {
		prints("makeslice: nel=");
		sys·printint(nel);
		prints("; cap=");
		sys·printint(cap);
		prints("; width=");
		sys·printint(width);
		prints("; ret=");
		sys·printslice(ret);
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

// sliceslice(old []any, lb int, hb int, width int) (ary []any);
void
sys·sliceslice(Slice old, uint32 lb, uint32 hb, uint32 width, Slice ret)
{

	if(hb > old.cap || lb > hb) {
		if(debug) {
			prints("sys·sliceslice: old=");
			sys·printslice(old);
			prints("; lb=");
			sys·printint(lb);
			prints("; hb=");
			sys·printint(hb);
			prints("; width=");
			sys·printint(width);
			prints("\n");

			prints("oldarray: nel=");
			sys·printint(old.len);
			prints("; cap=");
			sys·printint(old.cap);
			prints("\n");
		}
		throwslice(lb, hb, old.cap);
	}

	// new array is inside old array
	ret.len = hb-lb;
	ret.cap = old.cap - lb;
	ret.array = old.array + lb*width;

	FLUSH(&ret);

	if(debug) {
		prints("sys·sliceslice: old=");
		sys·printslice(old);
		prints("; lb=");
		sys·printint(lb);
		prints("; hb=");
		sys·printint(hb);
		prints("; width=");
		sys·printint(width);
		prints("; ret=");
		sys·printslice(ret);
		prints("\n");
	}
}

// slicearray(old *any, nel int, lb int, hb int, width int) (ary []any);
void
sys·slicearray(byte* old, uint32 nel, uint32 lb, uint32 hb, uint32 width, Slice ret)
{

	if(hb > nel || lb > hb) {
		if(debug) {
			prints("sys·slicearray: old=");
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
	ret.len = hb-lb;
	ret.cap = nel-lb;
	ret.array = old + lb*width;

	FLUSH(&ret);

	if(debug) {
		prints("sys·slicearray: old=");
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
		sys·printslice(ret);
		prints("\n");
	}
}

// arraytoslice(old *any, nel int) (ary []any)
void
sys·arraytoslice(byte* old, uint32 nel, Slice ret)
{

	// new dope to old array
	ret.len = nel;
	ret.cap = nel;
	ret.array = old;

	FLUSH(&ret);

	if(debug) {
		prints("sys·slicearrayp: old=");
		sys·printpointer(old);
		prints("; ret=");
		sys·printslice(ret);
		prints("\n");
	}
}

void
sys·printslice(Slice a)
{
	prints("[");
	sys·printint(a.len);
	prints("/");
	sys·printint(a.cap);
	prints("]");
	sys·printpointer(a.array);
}

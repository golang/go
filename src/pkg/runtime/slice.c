// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

static	int32	debug	= 0;

// makeslice(nel int, cap int, width int) (ary []any);
void
runtime·makeslice(uint32 nel, uint32 cap, uint32 width, Slice ret)
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
		runtime·printint(nel);
		prints("; cap=");
		runtime·printint(cap);
		prints("; width=");
		runtime·printint(width);
		prints("; ret=");
		runtime·printslice(ret);
		prints("\n");
	}
}

static void
throwslice(uint32 lb, uint32 hb, uint32 n)
{
	prints("slice[");
	runtime·printint(lb);
	prints(":");
	runtime·printint(hb);
	prints("] of [");
	runtime·printint(n);
	prints("] array\n");
	throw("array slice");
}

// sliceslice(old []any, lb int, hb int, width int) (ary []any);
void
runtime·sliceslice(Slice old, uint32 lb, uint32 hb, uint32 width, Slice ret)
{

	if(hb > old.cap || lb > hb) {
		if(debug) {
			prints("runtime·sliceslice: old=");
			runtime·printslice(old);
			prints("; lb=");
			runtime·printint(lb);
			prints("; hb=");
			runtime·printint(hb);
			prints("; width=");
			runtime·printint(width);
			prints("\n");

			prints("oldarray: nel=");
			runtime·printint(old.len);
			prints("; cap=");
			runtime·printint(old.cap);
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
		prints("runtime·sliceslice: old=");
		runtime·printslice(old);
		prints("; lb=");
		runtime·printint(lb);
		prints("; hb=");
		runtime·printint(hb);
		prints("; width=");
		runtime·printint(width);
		prints("; ret=");
		runtime·printslice(ret);
		prints("\n");
	}
}

// slicearray(old *any, nel int, lb int, hb int, width int) (ary []any);
void
runtime·slicearray(byte* old, uint32 nel, uint32 lb, uint32 hb, uint32 width, Slice ret)
{
	if(nel > 0 && old == nil) {
		// crash if old == nil.
		// could give a better message
		// but this is consistent with all the in-line checks
		// that the compiler inserts for other uses.
		*old = 0;
	}

	if(hb > nel || lb > hb) {
		if(debug) {
			prints("runtime·slicearray: old=");
			runtime·printpointer(old);
			prints("; nel=");
			runtime·printint(nel);
			prints("; lb=");
			runtime·printint(lb);
			prints("; hb=");
			runtime·printint(hb);
			prints("; width=");
			runtime·printint(width);
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
		prints("runtime·slicearray: old=");
		runtime·printpointer(old);
		prints("; nel=");
		runtime·printint(nel);
		prints("; lb=");
		runtime·printint(lb);
		prints("; hb=");
		runtime·printint(hb);
		prints("; width=");
		runtime·printint(width);
		prints("; ret=");
		runtime·printslice(ret);
		prints("\n");
	}
}

// arraytoslice(old *any, nel int) (ary []any)
void
runtime·arraytoslice(byte* old, uint32 nel, Slice ret)
{
	if(nel > 0 && old == nil) {
		// crash if old == nil.
		// could give a better message
		// but this is consistent with all the in-line checks
		// that the compiler inserts for other uses.
		*old = 0;
	}

	// new dope to old array
	ret.len = nel;
	ret.cap = nel;
	ret.array = old;

	FLUSH(&ret);

	if(debug) {
		prints("runtime·slicearrayp: old=");
		runtime·printpointer(old);
		prints("; ret=");
		runtime·printslice(ret);
		prints("\n");
	}
}

void
runtime·printslice(Slice a)
{
	prints("[");
	runtime·printint(a.len);
	prints("/");
	runtime·printint(a.cap);
	prints("]");
	runtime·printpointer(a.array);
}

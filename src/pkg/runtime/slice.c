// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "type.h"
#include "malloc.h"

static	int32	debug	= 0;

// see also unsafe·NewArray
// makeslice(typ *Type, nel int, cap int) (ary []any);
void
·makeslice(SliceType *t, uint32 nel, uint32 cap, Slice ret)
{
	uint64 size;

	if(cap < nel)
		cap = nel;
	size = cap*t->elem->size;

	ret.len = nel;
	ret.cap = cap;

	if((t->elem->kind&KindNoPointers))
		ret.array = mallocgc(size, RefNoPointers, 1);
	else
		ret.array = mal(size);

	FLUSH(&ret);

	if(debug) {
		printf("makeslice(%S, %d, %d); ret=", 
			*t->string, nel, cap);
 		·printslice(ret);
	}
}

static void
throwslice(uint32 lb, uint32 hb, uint32 n)
{
	prints("slice[");
	·printint(lb);
	prints(":");
	·printint(hb);
	prints("] of [");
	·printint(n);
	prints("] array\n");
	throw("array slice");
}

// sliceslice(old []any, lb int, hb int, width int) (ary []any);
void
·sliceslice(Slice old, uint32 lb, uint32 hb, uint32 width, Slice ret)
{
	if(hb > old.cap || lb > hb) {
		if(debug) {
			prints("runtime.sliceslice: old=");
			·printslice(old);
			prints("; lb=");
			·printint(lb);
			prints("; hb=");
			·printint(hb);
			prints("; width=");
			·printint(width);
			prints("\n");

			prints("oldarray: nel=");
			·printint(old.len);
			prints("; cap=");
			·printint(old.cap);
			prints("\n");
		}
		throwslice(lb, hb, old.cap);
	}

	// new array is inside old array
	ret.len = hb - lb;
	ret.cap = old.cap - lb;
	ret.array = old.array + lb*width;

	FLUSH(&ret);

	if(debug) {
		prints("runtime.sliceslice: old=");
		·printslice(old);
		prints("; lb=");
		·printint(lb);
		prints("; hb=");
		·printint(hb);
		prints("; width=");
		·printint(width);
		prints("; ret=");
		·printslice(ret);
		prints("\n");
	}
}

// sliceslice1(old []any, lb int, width int) (ary []any);
void
·sliceslice1(Slice old, uint32 lb, uint32 width, Slice ret)
{
	if(lb > old.len) {
		if(debug) {
			prints("runtime.sliceslice: old=");
			·printslice(old);
			prints("; lb=");
			·printint(lb);
			prints("; width=");
			·printint(width);
			prints("\n");

			prints("oldarray: nel=");
			·printint(old.len);
			prints("; cap=");
			·printint(old.cap);
			prints("\n");
		}
		throwslice(lb, old.len, old.cap);
	}

	// new array is inside old array
	ret.len = old.len - lb;
	ret.cap = old.cap - lb;
	ret.array = old.array + lb*width;

	FLUSH(&ret);

	if(debug) {
		prints("runtime.sliceslice: old=");
		·printslice(old);
		prints("; lb=");
		·printint(lb);
		prints("; width=");
		·printint(width);
		prints("; ret=");
		·printslice(ret);
		prints("\n");
	}
}

// slicearray(old *any, nel int, lb int, hb int, width int) (ary []any);
void
·slicearray(byte* old, uint32 nel, uint32 lb, uint32 hb, uint32 width, Slice ret)
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
			prints("runtime.slicearray: old=");
			·printpointer(old);
			prints("; nel=");
			·printint(nel);
			prints("; lb=");
			·printint(lb);
			prints("; hb=");
			·printint(hb);
			prints("; width=");
			·printint(width);
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
		prints("runtime.slicearray: old=");
		·printpointer(old);
		prints("; nel=");
		·printint(nel);
		prints("; lb=");
		·printint(lb);
		prints("; hb=");
		·printint(hb);
		prints("; width=");
		·printint(width);
		prints("; ret=");
		·printslice(ret);
		prints("\n");
	}
}

// slicecopy(to any, fr any, wid uint32) int
void
·slicecopy(Slice to, Slice fm, uintptr width, int32 ret)
{
	if(fm.array == nil || fm.len == 0 ||
	   to.array == nil || to.len == 0 ||
	   width == 0) {
		ret = 0;
		goto out;
	}

	ret = fm.len;
	if(to.len < ret)
		ret = to.len;

	if(ret == 1 && width == 1) {	// common case worth about 2x to do here
		*to.array = *fm.array;	// known to be a byte pointer
	} else {
		memmove(to.array, fm.array, ret*width);
	}

out:
	FLUSH(&ret);

	if(debug) {
		prints("main·copy: to=");
		·printslice(to);
		prints("; fm=");
		·printslice(fm);
		prints("; width=");
		·printint(width);
		prints("; ret=");
		·printint(ret);
		prints("\n");
	}
}

void
·printslice(Slice a)
{
	prints("[");
	·printint(a.len);
	prints("/");
	·printint(a.cap);
	prints("]");
	·printpointer(a.array);
}

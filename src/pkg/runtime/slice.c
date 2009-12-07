// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "type.h"
#include "malloc.h"

static	int32	debug	= 0;

// makeslice(typ *Type, nel int, cap int) (ary []any);
void
runtime·makeslice(SliceType *t, uint32 nel, uint32 cap, Slice ret)
{
	uint64 size;

	if(cap < nel)
		cap = nel;
	size = cap*t->elem->size;

	ret.len = nel;
	ret.cap = cap;

	// TODO(rsc): Disabled because reflect and gob cast []byte
	// to data structures with pointers.
	if(0 && (t->elem->kind&KindNoPointers))
		ret.array = mallocgc(size, RefNoPointers, 1);
	else
		ret.array = mal(size);

	FLUSH(&ret);

	if(debug) {
		printf("makeslice(%S, %d, %d); ret=", 
			*t->string, nel, cap);
 		runtime·printslice(ret);
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
	ret.len = hb - lb;
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

// sliceslice1(old []any, lb int, width int) (ary []any);
void
runtime·sliceslice1(Slice old, uint32 lb, uint32 width, Slice ret)
{
	if(lb > old.len) {
		if(debug) {
			prints("runtime·sliceslice: old=");
			runtime·printslice(old);
			prints("; lb=");
			runtime·printint(lb);
			prints("; width=");
			runtime·printint(width);
			prints("\n");

			prints("oldarray: nel=");
			runtime·printint(old.len);
			prints("; cap=");
			runtime·printint(old.cap);
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
		prints("runtime·sliceslice: old=");
		runtime·printslice(old);
		prints("; lb=");
		runtime·printint(lb);
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

// slicecopy(to any, fr any, wid uint32) int
void
runtime·slicecopy(Slice to, Slice fm, uintptr width, int32 ret)
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
		runtime·printslice(to);
		prints("; fm=");
		runtime·printslice(fm);
		prints("; width=");
		runtime·printint(width);
		prints("; ret=");
		runtime·printint(ret);
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

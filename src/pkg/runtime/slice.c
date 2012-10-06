// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "type.h"
#include "typekind.h"
#include "malloc.h"

static	bool	debug	= 0;

static	void	makeslice1(SliceType*, intgo, intgo, Slice*);
static	void	growslice1(SliceType*, Slice, intgo, Slice *);
	void	runtime·copy(Slice to, Slice fm, uintptr width, intgo ret);

// see also unsafe·NewArray
// makeslice(typ *Type, len, cap int64) (ary []any);
void
runtime·makeslice(SliceType *t, int64 len, int64 cap, Slice ret)
{
	if(len < 0 || (intgo)len != len)
		runtime·panicstring("makeslice: len out of range");
	
	if(cap < len || (intgo)cap != cap || t->elem->size > 0 && cap > MaxMem / t->elem->size)
		runtime·panicstring("makeslice: cap out of range");

	makeslice1(t, len, cap, &ret);

	if(debug) {
		runtime·printf("makeslice(%S, %D, %D); ret=",
			*t->string, len, cap);
		runtime·printslice(ret);
	}
}

// Dummy word to use as base pointer for make([]T, 0).
// Since you cannot take the address of such a slice,
// you can't tell that they all have the same base pointer.
static uintptr zerobase;

static void
makeslice1(SliceType *t, intgo len, intgo cap, Slice *ret)
{
	uintptr size;

	size = cap*t->elem->size;

	ret->len = len;
	ret->cap = cap;

	if(cap == 0)
		ret->array = (byte*)&zerobase;
	else if((t->elem->kind&KindNoPointers))
		ret->array = runtime·mallocgc(size, FlagNoPointers, 1, 1);
	else
		ret->array = runtime·mal(size);
}

// appendslice(type *Type, x, y, []T) []T
void
runtime·appendslice(SliceType *t, Slice x, Slice y, Slice ret)
{
	intgo m;
	uintptr w;

	m = x.len+y.len;

	if(m < x.len)
		runtime·throw("append: slice overflow");

	if(m > x.cap)
		growslice1(t, x, m, &ret);
	else
		ret = x;

	w = t->elem->size;
	runtime·memmove(ret.array + ret.len*w, y.array, y.len*w);
	ret.len += y.len;
	FLUSH(&ret);
}


// appendstr([]byte, string) []byte
void
runtime·appendstr(SliceType *t, Slice x, String y, Slice ret)
{
	intgo m;

	m = x.len+y.len;

	if(m < x.len)
		runtime·throw("append: slice overflow");

	if(m > x.cap)
		growslice1(t, x, m, &ret);
	else
		ret = x;

	runtime·memmove(ret.array + ret.len, y.str, y.len);
	ret.len += y.len;
	FLUSH(&ret);
}


// growslice(type *Type, x, []T, n int64) []T
void
runtime·growslice(SliceType *t, Slice old, int64 n, Slice ret)
{
	int64 cap;

	if(n < 1)
		runtime·panicstring("growslice: invalid n");

	cap = old.cap + n;

	if((intgo)cap != cap || cap < old.cap || (t->elem->size > 0 && cap > MaxMem/t->elem->size))
		runtime·panicstring("growslice: cap out of range");

	growslice1(t, old, cap, &ret);

	FLUSH(&ret);

	if(debug) {
		runtime·printf("growslice(%S,", *t->string);
		runtime·printslice(old);
		runtime·printf(", new cap=%D) =", cap);
		runtime·printslice(ret);
	}
}

static void
growslice1(SliceType *t, Slice x, intgo newcap, Slice *ret)
{
	intgo m;

	m = x.cap;
	
	// Using newcap directly for m+m < newcap handles
	// both the case where m == 0 and also the case where
	// m+m/4 wraps around, in which case the loop
	// below might never terminate.
	if(m+m < newcap)
		m = newcap;
	else {
		do {
			if(x.len < 1024)
				m += m;
			else
				m += m/4;
		} while(m < newcap);
	}
	makeslice1(t, x.len, m, ret);
	runtime·memmove(ret->array, x.array, ret->len * t->elem->size);
}

// copy(to any, fr any, wid uintptr) int
void
runtime·copy(Slice to, Slice fm, uintptr width, intgo ret)
{
	if(fm.len == 0 || to.len == 0 || width == 0) {
		ret = 0;
		goto out;
	}

	ret = fm.len;
	if(to.len < ret)
		ret = to.len;

	if(ret == 1 && width == 1) {	// common case worth about 2x to do here
		*to.array = *fm.array;	// known to be a byte pointer
	} else {
		runtime·memmove(to.array, fm.array, ret*width);
	}

out:
	FLUSH(&ret);

	if(debug) {
		runtime·prints("main·copy: to=");
		runtime·printslice(to);
		runtime·prints("; fm=");
		runtime·printslice(fm);
		runtime·prints("; width=");
		runtime·printint(width);
		runtime·prints("; ret=");
		runtime·printint(ret);
		runtime·prints("\n");
	}
}

void
runtime·slicestringcopy(Slice to, String fm, intgo ret)
{
	if(fm.len == 0 || to.len == 0) {
		ret = 0;
		goto out;
	}

	ret = fm.len;
	if(to.len < ret)
		ret = to.len;

	runtime·memmove(to.array, fm.str, ret);

out:
	FLUSH(&ret);
}

void
runtime·printslice(Slice a)
{
	runtime·prints("[");
	runtime·printint(a.len);
	runtime·prints("/");
	runtime·printint(a.cap);
	runtime·prints("]");
	runtime·printpointer(a.array);
}

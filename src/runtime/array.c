// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

static	int32	debug	= 0;

// newarray(nel uint32, cap uint32, width uint32) (ary *[]any);
void
sys·newarray(uint32 nel, uint32 cap, uint32 width, Array* ret)
{
	Array *d;
	uint64 size;

	if(cap < nel)
		cap = nel;
	size = cap*width;

	d = mal(sizeof(*d) - sizeof(d->b) + size);
	d->nel = nel;
	d->cap = cap;
	d->array = d->b;

	ret = d;
	FLUSH(&d);

	if(debug) {
		prints("newarray: nel=");
		sys·printint(nel);
		prints("; cap=");
		sys·printint(cap);
		prints("; width=");
		sys·printint(width);
		prints("; ret=");
		sys·printpointer(ret);
		prints("\n");
	}
}

// arraysliced(old *[]any, lb uint32, hb uint32, width uint32) (ary *[]any);
void
sys·arraysliced(Array* old, uint32 lb, uint32 hb, uint32 width, Array* ret)
{
	Array *d;

	if(hb > old->cap || lb > hb) {
		if(debug) {
			prints("sys·arrayslices: old=");
			sys·printpointer(old);
			prints("; lb=");
			sys·printint(lb);
			prints("; hb=");
			sys·printint(hb);
			prints("; width=");
			sys·printint(width);
			prints("\n");

			prints("oldarray: nel=");
			sys·printint(old->nel);
			prints("; cap=");
			sys·printint(old->cap);
			prints("\n");
		}
		throw("sys·arraysliced: new size exceeds old size");
	}

	// new array is inside old array
	d = mal(sizeof(*d) - sizeof(d->b));
	d->nel = hb-lb;
	d->cap = old->cap - lb;
	d->array = old->array + lb*width;

	ret = d;
	FLUSH(&d);

	if(debug) {
		prints("sys·arrayslices: old=");
		sys·printpointer(old);
		prints("; lb=");
		sys·printint(lb);
		prints("; hb=");
		sys·printint(hb);
		prints("; width=");
		sys·printint(width);
		prints("; ret=");
		sys·printpointer(ret);
		prints("\n");
	}
}

// arrayslices(old *any, nel uint32, lb uint32, hb uint32, width uint32) (ary *[]any);
void
sys·arrayslices(byte* old, uint32 nel, uint32 lb, uint32 hb, uint32 width, Array* ret)
{
	Array *d;

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
		throw("sys·arrayslices: new size exceeds cap");
	}

	// new array is inside old array
	d = mal(sizeof(*d) - sizeof(d->b));
	d->nel = hb-lb;
	d->cap = nel-lb;
	d->array = old + lb*width;

	ret = d;
	FLUSH(&d);

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
		sys·printpointer(ret);
		prints("\n");
	}
}

// arrays2d(old *any, nel uint32) (ary *[]any)
void
sys·arrays2d(byte* old, uint32 nel, Array* ret)
{
	Array *d;

	// new dope to old array
	d = mal(sizeof(*d) - sizeof(d->b));
	d->nel = nel;
	d->cap = nel;
	d->array = old;

	ret = d;
	FLUSH(&d);

	if(debug) {
		prints("sys·arrays2d: old=");
		sys·printpointer(old);
		prints("; nel=");
		sys·printint(nel);
		prints("; ret=");
		sys·printpointer(ret);
		prints("\n");
	}
}

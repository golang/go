// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

static	int32	debug	= 0;

typedef	struct	Sigt	Sigt;
typedef	struct	Sigi	Sigi;
typedef	struct	Map	Map;

struct	Sigt
{
	byte*	name;
	uint32	hash;
	uint32	offset;		// offset of substruct
	uint32	width;		// width of type
	uint32	elemalg;	// algorithm of type
	void	(*fun)(void);
};

struct	Sigi
{
	byte*	name;
	uint32	hash;
	uint32	perm;		// location of fun in Sigt
};

struct	Map
{
	Sigi*	sigi;
	Sigt*	sigt;
	Map*	link;
	int32	bad;
	int32	unused;
	void	(*fun[])(void);
};

static	Map*	hash[1009];

static void
printsigi(Sigi *si)
{
	int32 i, n;
	byte *name;

	sys·printpointer(si);
	prints("{");
	n = si[0].perm;		// first entry has size
	for(i=1; i<n; i++) {
		name = si[i].name;
		if(name == nil) {
			prints("<nil>");
			break;
		}
		prints("[");
		sys·printint(i);
		prints("]\"");
		prints((int8*)name);
		prints("\"");
		sys·printint(si[i].hash%999);
		prints("/");
		sys·printint(si[i].perm);
	}
	prints("}");
}

static void
printsigt(Sigt *st)
{
	int32 i;
	byte *name;

	sys·printpointer(st);
	prints("{");
	for(i=0;; i++) {
		name = st[i].name;
		if(name == nil)
			break;
		prints("[");
		sys·printint(i);
		prints("]\"");
		prints((int8*)name);
		prints("\"");
		sys·printint(st[i].hash%999);
		prints("/");
		sys·printint(st[i].offset);
		prints(",");
		sys·printint(st[i].width);
		prints(",");
		sys·printint(st[i].elemalg);
		prints("/");
		sys·printpointer(st[i].fun);
	}
	prints("}");
}

static void
printiface(Map *im, void *it)
{
	prints("(");
	sys·printpointer(im);
	prints(",");
	sys·printpointer(it);
	prints(")");
}

static Map*
hashmap(Sigi *si, Sigt *st)
{
	int32 nt, ni;
	uint32 ihash, h;
	byte *sname, *iname;
	Map *m;

	h = ((uint32)(uint64)si + (uint32)(uint64)st) % nelem(hash);
	for(m=hash[h]; m!=nil; m=m->link) {
		if(m->sigi == si && m->sigt == st) {
			if(m->bad) {
				throw("bad hashmap");
				m = nil;
			}
			// prints("old hashmap\n");
			return m;
		}
	}

	ni = si[0].perm;	// first entry has size
	m = mal(sizeof(*m) + ni*sizeof(m->fun[0]));
	m->sigi = si;
	m->sigt = st;

	ni = 1;			// skip first word
	nt = 0;

loop1:
	// pick up next name from
	// interface signature
	iname = si[ni].name;
	if(iname == nil) {
		m->link = hash[h];
		hash[h] = m;
		// prints("new hashmap\n");
		return m;
	}
	ihash = si[ni].hash;

loop2:
	// pick up and comapre next name
	// from structure signature
	sname = st[nt].name;
	if(sname == nil) {
		prints((int8*)iname);
		prints(": ");
		throw("hashmap: failed to find method");
		m->bad = 1;
		m->link = hash[h];
		hash[h] = m;
		return nil;
	}

	if(ihash != st[nt].hash ||
	   strcmp(sname, iname) != 0) {
		nt++;
		goto loop2;
	}

	m->fun[si[ni].perm] = st[nt].fun;
	ni++;
	goto loop1;
}

// ifaceT2I(sigi *byte, sigt *byte, elem any) (ret any);
void
sys·ifaceT2I(Sigi *si, Sigt *st, void *elem, Map *retim, void *retit)
{

	if(debug) {
		prints("T2I sigi=");
		printsigi(si);
		prints(" sigt=");
		printsigt(st);
		prints(" elem=");
		sys·printpointer(elem);
		prints("\n");
	}

	retim = hashmap(si, st);
	retit = elem;

	if(debug) {
		prints("T2I ret=");
		printiface(retim, retit);
		prints("\n");
	}

	FLUSH(&retim);
}

// ifaceI2T(sigt *byte, iface any) (ret any);
void
sys·ifaceI2T(Sigt *st, Map *im, void *it, void *ret)
{

	if(debug) {
		prints("I2T sigt=");
		printsigt(st);
		prints(" iface=");
		printiface(im, it);
		prints("\n");
	}

	if(im == nil)
		throw("ifaceI2T: nil map");

	if(im->sigt != st)
		throw("ifaceI2T: wrong type");

	ret = it;
	if(debug) {
		prints("I2T ret=");
		sys·printpointer(ret);
		prints("\n");
	}

	FLUSH(&ret);
}

// ifaceI2I(sigi *byte, iface any) (ret any);
void
sys·ifaceI2I(Sigi *si, Map *im, void *it, Map *retim, void *retit)
{

	if(debug) {
		prints("I2I sigi=");
		printsigi(si);
		prints(" iface=");
		printiface(im, it);
		prints("\n");
	}

	if(im == nil) {
		throw("ifaceI2I: nil map");
		return;
	}

	retit = it;
	retim = im;
	if(im->sigi != si)
		retim = hashmap(si, im->sigt);

	if(debug) {
		prints("I2I ret=");
		printiface(retim, retit);
		prints("\n");
	}

	FLUSH(&retim);
}

void
sys·printinter(Map *im, void *it)
{
	printiface(im, it);
}

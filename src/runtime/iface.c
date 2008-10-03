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
	for(i=1;; i++) {
		name = si[i].name;
		if(name == nil)
			break;
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
	sys·printint(st[0].hash);	// first element has alg
	prints(",");
	sys·printint(st[0].offset);	// first element has width
	for(i=1;; i++) {
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

	nt = 0;
	for(ni=1; (iname=si[ni].name) != nil; ni++) {	// ni=1: skip first word
		// pick up next name from
		// interface signature
		ihash = si[ni].hash;

		for(;; nt++) {
			// pick up and compare next name
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
			if(ihash == st[nt].hash && strcmp(sname, iname) == 0)
				break;
		}
		m->fun[si[ni].perm] = st[nt].fun;
	}
	m->link = hash[h];
	hash[h] = m;
	// prints("new hashmap\n");
	return m;
}

// ifaceT2I(sigi *byte, sigt *byte, elem any) (ret any);
void
sys·ifaceT2I(Sigi *si, Sigt *st, void *elem, Map *retim, void *retit)
{
//	int32 alg, wid;

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

//	alg = st->hash;
//	wid = st->offset;
//	algarray[alg].copy(wid, &retit, &elem);
	retit = elem;		// for speed could do this

	if(debug) {
		prints("T2I ret=");
		printiface(retim, retit);
		prints("\n");
	}

	FLUSH(&retim);
	FLUSH(&retit);
}

// ifaceI2T(sigt *byte, iface any) (ret any);
void
sys·ifaceI2T(Sigt *st, Map *im, void *it, void *ret)
{
//	int32 alg, wid;

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

//	alg = st->hash;
//	wid = st->offset;
//	algarray[alg].copy(wid, &ret, &it);
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
	FLUSH(&retit);
}

void
sys·printinter(Map *im, void *it)
{
	printiface(im, it);
}

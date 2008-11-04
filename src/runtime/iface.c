// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

static	int32	debug	= 0;

enum
{
	ASIMP		= 0,
	ASTRING,
	APTR,
	AINTER,
};

typedef	struct	Sigt	Sigt;
typedef	struct	Sigi	Sigi;
typedef	struct	Map	Map;

struct	Sigt
{
	byte*	name;
	uint32	hash;		// hash of type		// first is alg
	uint32	offset;		// offset of substruct	// first is width
	void	(*fun)(void);
};

struct	Sigi
{
	byte*	name;
	uint32	hash;
	uint32	perm;		// location of fun in Sigt // first is size
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

#define	END	nil,0,0,nil

Sigi	sigi·inter[2] =	{ (byte*)"interface {}", 0, 0, nil, 0, 0 };

Sigt	sigt·int8[2] =	{ (byte*)"int8", ASIMP, 1, nil, END };
Sigt	sigt·int16[2] =	{ (byte*)"int16", ASIMP, 2, nil, END };
Sigt	sigt·int32[2] =	{ (byte*)"int32", ASIMP, 4, nil, END };
Sigt	sigt·int64[2] =	{ (byte*)"int64", ASIMP, 8, nil, END };

Sigt	sigt·uint8[2] =	{ (byte*)"uint8", ASIMP, 1, nil, END };
Sigt	sigt·uint16[2] =	{ (byte*)"uint16", ASIMP, 2, nil, END };
Sigt	sigt·uint32[2] =	{ (byte*)"uint32", ASIMP, 4, nil, END };
Sigt	sigt·uint64[2] =	{ (byte*)"uint64", ASIMP, 8, nil, END };

Sigt	sigt·float32[2] =	{ (byte*)"float32", ASIMP, 4, nil, END };
Sigt	sigt·float64[2] =	{ (byte*)"float64", ASIMP, 8, nil, END };
//Sigt	sigt·float80[2] =	{ (byte*)"float80", ASIMP, 0, nil, END };

Sigt	sigt·bool[2] =	{ (byte*)"bool", ASIMP, 1, nil, END };
Sigt	sigt·string[2] =	{ (byte*)"string", ASTRING, 8, nil, END };

Sigt	sigt·int[2] =	{ (byte*)"int", ASIMP, 4, nil, END };
Sigt	sigt·uint[2] =	{ (byte*)"uint", ASIMP, 4, nil, END };
Sigt	sigt·uintptr[2] =	{ (byte*)"uintptr", ASIMP, 8, nil, END };
Sigt	sigt·float[2] =	{ (byte*)"float", ASIMP, 4, nil, END };

static void
printsigi(Sigi *si)
{
	int32 i;
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
hashmap(Sigi *si, Sigt *st, int32 canfail)
{
	int32 nt, ni;
	uint32 ihash, h;
	byte *sname, *iname;
	Map *m;

	h = ((uint32)(uint64)si + (uint32)(uint64)st) % nelem(hash);
	for(m=hash[h]; m!=nil; m=m->link) {
		if(m->sigi == si && m->sigt == st) {
			if(m->bad) {
				if(!canfail)
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

	nt = 1;
	for(ni=1;; ni++) {	// ni=1: skip first word
		iname = si[ni].name;
		if(iname == nil)
			break;

		// pick up next name from
		// interface signature
		ihash = si[ni].hash;

		for(;; nt++) {
			// pick up and compare next name
			// from structure signature
			sname = st[nt].name;
			if(sname == nil) {
				if(!canfail) {
					prints("cannot convert type ");
					prints((int8*)st[0].name);
					prints(" to interface ");
					prints((int8*)si[0].name);
					prints(": missing method ");
					prints((int8*)iname);
					prints("\n");
					throw("interface conversion");
				}
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

	retim = hashmap(si, st, 0);

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
		// If incoming interface is uninitialized (zeroed)
		// make the outgoing interface zeroed as well.
		retim = nil;
		retit = nil;
	} else {
		retit = it;
		retim = im;
		if(im->sigi != si)
			retim = hashmap(si, im->sigt, 0);
	}

	if(debug) {
		prints("I2I ret=");
		printiface(retim, retit);
		prints("\n");
	}

	FLUSH(&retim);
	FLUSH(&retit);
}

// ifaceeq(i1 any, i2 any) (ret bool);
void
sys·ifaceeq(Map *im1, void *it1, Map *im2, void *it2, byte ret)
{
	int32 alg, wid;

	if(debug) {
		prints("Ieq i1=");
		printiface(im1, it1);
		prints(" i2=");
		printiface(im2, it2);
		prints("\n");
	}

	ret = false;

	// are they both nil
	if(im1 == nil) {
		if(im2 == nil)
			goto yes;
		goto no;
	}
	if(im2 == nil)
		goto no;

	// value
	alg = im1->sigt->hash;
	if(alg != im2->sigt->hash)
		goto no;

	wid = im1->sigt->offset;
	if(wid != im2->sigt->offset)
		goto no;

	if(!algarray[alg].equal(wid, &it1, &it2))
		goto no;

yes:
	ret = true;
no:
	if(debug) {
		prints("Ieq ret=");
		sys·printbool(ret);
		prints("\n");
	}
	FLUSH(&ret);
}

void
sys·printinter(Map *im, void *it)
{
	printiface(im, it);
}

void
sys·reflect(Map *im, void *it, uint64 retit, string rettype)
{
	string s;
	int32 n;
	byte *type;

	if(im == nil) {
		retit = 0;
		rettype = nil;
	} else {
		retit = (uint64)it;
		type = im->sigt->name;
		n = findnull((int8*)type);
		s = mal(sizeof *s + n + 1);
		s->len = n;
		mcpy(s->str, type, n);
		rettype = s;
	}
	FLUSH(&retit);
	FLUSH(&rettype);
}

extern Sigt *gotypesigs[];
extern int32 ngotypesigs;

static Sigt*
fakesigt(string type)
{
	// TODO(rsc): Cache these by type string.
	Sigt *sigt;

	// Must be pointer in order for alg, width to be right.
	if(type == nil || type->len == 0 || type->str[0] != '*') {
		// TODO(rsc): What to do here?
		prints("bad unreflect type: ");
		sys·printstring(type);
		prints("\n");
		throw("unreflect");
	}
	sigt = mal(2*sizeof sigt[0]);
	sigt[0].name = mal(type->len + 1);
	mcpy(sigt[0].name, type->str, type->len);
	sigt[0].hash = ASIMP;	// alg
	sigt[0].offset = sizeof(void*);	// width
	return sigt;
}

static int32
cmpstringchars(string a, uint8 *b)
{
	int32 i;

	for(i=0;; i++) {
		if(i == a->len) {
			if(b[i] == 0)
				return 0;
			return -1;
		}
		if(b[i] == 0)
			return 1;
		if(a->str[i] != b[i]) {
			if((uint8)a->str[i] < (uint8)b[i])
				return -1;
			return 1;
		}
	}
}

static Sigt*
findtype(string type)
{
	int32 i;

	for(i=0; i<ngotypesigs; i++)
		if(cmpstringchars(type, gotypesigs[i]->name) == 0)
			return gotypesigs[i];
	return fakesigt(type);
}

void
sys·unreflect(uint64 it, string type, Map *retim, void *retit)
{
	if(cmpstring(type, emptystring) == 0) {
		retim = 0;
		retit = 0;
	} else {
		retim = hashmap(sigi·inter, findtype(type), 0);
		retit = (void*)it;
	}
	FLUSH(&retim);
	FLUSH(&retit);
}


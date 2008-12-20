// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

static	int32	debug	= 0;

typedef	struct	Sigt	Sigt;
typedef	struct	Sigi	Sigi;
typedef	struct	Itype	Itype;

/*
 * the layout of Iface, Sigt and Sigi are known to the compiler
 */
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

struct	Itype
{
	Sigi*	sigi;
	Sigt*	sigt;
	Itype*	link;
	int32	bad;
	int32	unused;
	void	(*fun[])(void);
};

static	Itype*	hash[1009];

Sigi	sigi·empty[2] =	{ (byte*)"interface { }" };

static void
printsigi(Sigi *si)
{
	int32 i;
	byte *name;

	sys·printpointer(si);
	prints("{");
	prints((int8*)si[0].name);
	prints(":");
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
	prints((int8*)st[0].name);
	prints(":");
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
printiface(Iface i)
{
	int32 j;

	prints("(");
	sys·printpointer(i.type);
	prints(",");
	for(j=0; j<nelem(i.data); j++) {
		if(j > 0)
			prints(".");
		sys·printpointer(i.data[0]);
	}
	prints(")");
}

static Itype*
itype(Sigi *si, Sigt *st, int32 canfail)
{
	int32 nt, ni;
	uint32 ihash, h;
	byte *sname, *iname;
	Itype *m;

	h = ((uint32)(uint64)si + (uint32)(uint64)st) % nelem(hash);
	for(m=hash[h]; m!=nil; m=m->link) {
		if(m->sigi == si && m->sigt == st) {
			if(m->bad) {
				m = nil;
				if(!canfail) {
					// this can only happen if the conversion
					// was already done once using the , ok form
					// and we have a cached negative result.
					// the cached result doesn't record which
					// interface function was missing, so jump
					// down to the interface check, which will
					// give a better error.
					goto throw;
				}
			}
			// prints("old itype\n");
			return m;
		}
	}

	ni = si[0].perm;	// first entry has size
	m = mal(sizeof(*m) + ni*sizeof(m->fun[0]));
	m->sigi = si;
	m->sigt = st;

throw:
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
	// prints("new itype\n");
	return m;
}

// ifaceT2I(sigi *byte, sigt *byte, elem any) (ret any);
void
sys·ifaceT2I(Sigi *si, Sigt *st, ...)
{
	byte *elem;
	Iface *ret;
	int32 alg, wid;

	elem = (byte*)(&st+1);
	wid = st->offset;
	ret = (Iface*)(elem + rnd(wid, 8));
	ret->type = itype(si, st, 0);

	if(debug) {
		prints("T2I sigi=");
		printsigi(si);
		prints(" sigt=");
		printsigt(st);
		prints(" elem=");
		sys·printpointer(*(void**)elem);
		prints("\n");
	}

	alg = st->hash;
	wid = st->offset;
	if(wid <= sizeof ret->data)
		algarray[alg].copy(wid, ret->data, elem);
	else{
		ret->data[0] = mal(wid);
		if(debug)
			printf("T2I mal %d %p\n", wid, ret->data[0]);
		algarray[alg].copy(wid, ret->data[0], elem);
	}

	if(debug) {
		prints("T2I ret=");
		printiface(*ret);
		prints("\n");
	}

	FLUSH(&ret);
}

// ifaceI2T(sigt *byte, iface any) (ret any);
void
sys·ifaceI2T(Sigt *st, Iface i, ...)
{
	Itype *im;
	byte *ret;
	int32 wid, alg;

	ret = (byte*)(&i+1);

	if(debug) {
		prints("I2T sigt=");
		printsigt(st);
		prints(" iface=");
		printiface(i);
		prints("\n");
	}

	im = i.type;
	if(im == nil) {
		prints("interface is nil, not ");
		prints((int8*)st[0].name);
		prints("\n");
		throw("interface conversion");
	}

	if(im->sigt != st) {
		prints((int8*)im->sigi[0].name);
		prints(" is ");
		prints((int8*)im->sigt[0].name);
		prints(", not ");
		prints((int8*)st[0].name);
		prints("\n");
		throw("interface conversion");
	}

	alg = st->hash;
	wid = st->offset;
	if(wid <= sizeof i.data)
		algarray[alg].copy(wid, ret, i.data);
	else
		algarray[alg].copy(wid, ret, i.data[0]);

	if(debug) {
		prints("I2T ret=");
		sys·printpointer(*(void**)ret);
		prints("\n");
	}
	FLUSH(&ret);
}

// ifaceI2T2(sigt *byte, iface any) (ret any, ok bool);
void
sys·ifaceI2T2(Sigt *st, Iface i, ...)
{
	byte *ret;
	bool *ok;
	Itype *im;
	int32 alg, wid;

	ret = (byte*)(&i+1);
	alg = st->hash;
	wid = st->offset;
	ok = (bool*)(ret+rnd(wid, 8));

	if(debug) {
		prints("I2T2 sigt=");
		printsigt(st);
		prints(" iface=");
		printiface(i);
		prints("\n");
	}

	im = i.type;
	if(im == nil || im->sigt != st) {
		*ok = false;
		sys·memclr(ret, wid);
	} else {
		*ok = true;
		if(wid <= sizeof i.data)
			algarray[alg].copy(wid, ret, i.data);
		else
			algarray[alg].copy(wid, ret, i.data[0]);
	}
	if(debug) {
		prints("I2T2 ret=");
		sys·printpointer(*(void**)ret);
		sys·printbool(*ok);
		prints("\n");
	}
}

// ifaceI2I(sigi *byte, iface any) (ret any);
void
sys·ifaceI2I(Sigi *si, Iface i, Iface ret)
{
	Itype *im;
	int32 j;

	if(debug) {
		prints("I2I sigi=");
		printsigi(si);
		prints(" iface=");
		printiface(i);
		prints("\n");
	}

	im = i.type;
	if(im == nil) {
		// If incoming interface is uninitialized (zeroed)
		// make the outgoing interface zeroed as well.
		ret.type = nil;
		for(j=0; j<nelem(ret.data); j++)
			ret.data[j] = nil;
	} else {
		ret = i;
		if(im->sigi != si)
			ret.type = itype(si, im->sigt, 0);
	}

	if(debug) {
		prints("I2I ret=");
		printiface(ret);
		prints("\n");
	}

	FLUSH(&ret);
}

// ifaceI2I2(sigi *byte, iface any) (ret any, ok bool);
void
sys·ifaceI2I2(Sigi *si, Iface i, Iface ret, bool ok)
{
	Itype *im;
	int32 j;

	if(debug) {
		prints("I2I2 sigi=");
		printsigi(si);
		prints(" iface=");
		printiface(i);
		prints("\n");
	}

	im = i.type;
	if(im == nil) {
		// If incoming interface is uninitialized (zeroed)
		// make the outgoing interface zeroed as well.
		ret.type = nil;
		for(j=0; j<nelem(ret.data); j++)
			ret.data[j] = nil;
		ok = 1;
	} else {
		ret = i;
		ok = 1;
		if(im->sigi != si) {
			ret.type = itype(si, im->sigt, 1);
			if(ret.type == nil) {
				for(j=0; j<nelem(ret.data); j++)
					ret.data[j] = nil;
				ok = 0;
			}
		}
	}

	if(debug) {
		prints("I2I ret=");
		printiface(ret);
		prints("\n");
	}

	FLUSH(&ret);
	FLUSH(&ok);
}

// ifaceeq(i1 any, i2 any) (ret bool);
void
sys·ifaceeq(Iface i1, Iface i2, bool ret)
{
	int32 alg, wid;

	if(debug) {
		prints("Ieq i1=");
		printiface(i1);
		prints(" i2=");
		printiface(i2);
		prints("\n");
	}

	ret = false;

	// are they both nil
	if(i1.type == nil) {
		if(i2.type == nil)
			goto yes;
		goto no;
	}
	if(i2.type == nil)
		goto no;

	// value
	alg = i1.type->sigt->hash;
	if(alg != i2.type->sigt->hash)
		goto no;

	wid = i1.type->sigt->offset;
	if(wid != i2.type->sigt->offset)
		goto no;

	if(wid <= sizeof i1.data) {
		if(!algarray[alg].equal(wid, i1.data, i2.data))
			goto no;
	} else {
		if(!algarray[alg].equal(wid, i1.data[0], i2.data[0]))
			goto no;
	}

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
sys·printinter(Iface i)
{
	printiface(i);
}

void
sys·reflect(Itype *im, void *it, uint64 retit, string rettype)
{
	if(im == nil) {
		retit = 0;
		rettype = nil;
	} else {
		retit = (uint64)it;
		rettype = gostring(im->sigt->name);
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
sys·unreflect(uint64 it, string type, Itype *retim, void *retit)
{
	if(cmpstring(type, emptystring) == 0) {
		retim = 0;
		retit = 0;
	} else {
		retim = itype(sigi·empty, findtype(type), 0);
		retit = (void*)it;
	}
	FLUSH(&retim);
	FLUSH(&retit);
}


// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

int32	iface_debug	= 0;

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

static	Iface	niliface;
static	Itype*	hash[1009];
static	Lock	ifacelock;

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
	prints("(");
	sys·printpointer(i.type);
	prints(",");
	sys·printpointer(i.data);
	prints(")");
}

static Itype*
itype(Sigi *si, Sigt *st, int32 canfail)
{
	int32 locked;
	int32 nt, ni;
	uint32 ihash, h;
	byte *sname, *iname;
	Itype *m;

	// compiler has provided some good hash codes for us.
	h = 0;
	if(si)
		h += si->hash;
	if(st)
		h += st->hash >> 8;
	h %= nelem(hash);

	// look twice - once without lock, once with.
	// common case will be no lock contention.
	for(locked=0; locked<2; locked++) {
		if(locked)
			lock(&ifacelock);
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
				if(locked)
					unlock(&ifacelock);
				return m;
			}
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
					printf("cannot convert type %s to interface %s: missing method %s\n",
						st[0].name, si[0].name, iname);
					if(iface_debug) {
						prints("interface");
						printsigi(si);
						prints("\ntype");
						printsigt(st);
						prints("\n");
					}
					throw("interface conversion");
				}
				m->bad = 1;
				m->link = hash[h];
				hash[h] = m;
				if(locked)
					unlock(&ifacelock);
				return nil;
			}
			if(ihash == st[nt].hash && strcmp(sname, iname) == 0)
				break;
		}
		m->fun[si[ni].perm] = st[nt].fun;
	}
	m->link = hash[h];
	hash[h] = m;
	// printf("new itype %p\n", m);
	if(locked)
		unlock(&ifacelock);
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

	if(iface_debug) {
		prints("T2I sigi=");
		printsigi(si);
		prints(" sigt=");
		printsigt(st);
		prints(" elem=");
		sys·printpointer(*(void**)elem);
		prints("\n");
	}

	alg = st->hash & 0xFF;
	wid = st->offset;
	if(wid <= sizeof ret->data)
		algarray[alg].copy(wid, &ret->data, elem);
	else{
		ret->data = mal(wid);
		if(iface_debug)
			printf("T2I mal %d %p\n", wid, ret->data);
		algarray[alg].copy(wid, ret->data, elem);
	}

	if(iface_debug) {
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

	if(iface_debug) {
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

	alg = st->hash & 0xFF;
	wid = st->offset;
	if(wid <= sizeof i.data)
		algarray[alg].copy(wid, ret, &i.data);
	else
		algarray[alg].copy(wid, ret, i.data);

	if(iface_debug) {
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
	alg = st->hash & 0xFF;
	wid = st->offset;
	ok = (bool*)(ret+rnd(wid, 1));

	if(iface_debug) {
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
			algarray[alg].copy(wid, ret, &i.data);
		else
			algarray[alg].copy(wid, ret, i.data);
	}
	if(iface_debug) {
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

	if(iface_debug) {
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
		ret = niliface;
	} else {
		ret = i;
		if(im->sigi != si)
			ret.type = itype(si, im->sigt, 0);
	}

	if(iface_debug) {
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

	if(iface_debug) {
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
		ret = niliface;
		ok = 1;
	} else {
		ret = i;
		ok = 1;
		if(im->sigi != si) {
			ret.type = itype(si, im->sigt, 1);
			if(ret.type == nil) {
				ret = niliface;
				ok = 0;
			}
		}
	}

	if(iface_debug) {
		prints("I2I ret=");
		printiface(ret);
		prints("\n");
	}

	FLUSH(&ret);
	FLUSH(&ok);
}

uint64
ifacehash(Iface a)
{
	int32 alg, wid;

	if(a.type == nil)
		return 0;
	alg = a.type->sigt->hash & 0xFF;
	wid = a.type->sigt->offset;
	if(algarray[alg].hash == nohash) {
		// calling nohash will throw too,
		// but we can print a better error.
		printf("hash of unhashable type %s\n", a.type->sigt->name);
		throw("interface hash");
	}
	if(wid <= sizeof a.data)
		return algarray[alg].hash(wid, &a.data);
	else
		return algarray[alg].hash(wid, a.data);
}

bool
ifaceeq(Iface i1, Iface i2)
{
	int32 alg, wid;
	bool ret;

	if(iface_debug) {
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

	// are they the same type?
	if(i1.type->sigt != i2.type->sigt)
		goto no;

	alg = i1.type->sigt->hash & 0xFF;
	wid = i1.type->sigt->offset;

	if(algarray[alg].equal == noequal) {
		// calling noequal will throw too,
		// but we can print a better error.
		printf("comparing uncomparable type %s\n", i1.type->sigt->name);
		throw("interface compare");
	}

	if(wid <= sizeof i1.data) {
		if(!algarray[alg].equal(wid, &i1.data, &i2.data))
			goto no;
	} else {
		if(!algarray[alg].equal(wid, i1.data, i2.data))
			goto no;
	}

yes:
	ret = true;
no:
	if(iface_debug) {
		prints("Ieq ret=");
		sys·printbool(ret);
		prints("\n");
	}
	return ret;
}

// ifaceeq(i1 any, i2 any) (ret bool);
void
sys·ifaceeq(Iface i1, Iface i2, bool ret)
{
	ret = ifaceeq(i1, i2);
	FLUSH(&ret);
}

void
sys·printinter(Iface i)
{
	printiface(i);
}

void
sys·Reflect(Iface i, uint64 retit, string rettype, bool retindir)
{
	int32 wid;

	if(i.type == nil) {
		retit = 0;
		rettype = nil;
		retindir = false;
	} else {
		retit = (uint64)i.data;
		rettype = gostring(i.type->sigt->name);
		wid = i.type->sigt->offset;
		retindir = wid > sizeof i.data;
	}
	FLUSH(&retit);
	FLUSH(&rettype);
	FLUSH(&retindir);
}

extern Sigt *gotypesigs[];
extern int32 ngotypesigs;


// The reflection library can ask to unreflect on a type
// that has never been used, so we don't have a signature for it.
// For concreteness, suppose a program does
//
// 	type T struct{ x []int }
// 	var t T;
// 	v := reflect.NewValue(v);
// 	vv := v.Field(0);
// 	if s, ok := vv.Interface().(string) {
// 		print("first field is string");
// 	}
//
// vv.Interface() returns the result of sys.Unreflect with
// a typestring of "[]int".  If []int is not used with interfaces
// in the rest of the program, there will be no signature in gotypesigs
// for "[]int", so we have to invent one.  The only requirements
// on the fake signature are:
//
//	(1) any interface conversion using the signature will fail
//	(2) calling sys.Reflect() returns the args to unreflect
//
// (1) is ensured by the fact that we allocate a new Sigt,
// so it will necessarily be != any Sigt in gotypesigs.
// (2) is ensured by storing the type string in the signature
// and setting the width to force the correct value of the bool indir.
//
// Note that (1) is correct behavior: if the program had tested
// for .([]int) instead of .(string) above, then there would be a
// signature with type string "[]int" in gotypesigs, and unreflect
// wouldn't call fakesigt.

static	Sigt	*fake[1009];
static	int32	nfake;

static Sigt*
fakesigt(string type, bool indir)
{
	Sigt *sigt;
	uint32 h;
	int32 i, locked;

	if(type == nil)
		type = emptystring;

	h = 0;
	for(i=0; i<type->len; i++)
		h = h*37 + type->str[i];
	h += indir;
	h %= nelem(fake);

	for(locked=0; locked<2; locked++) {
		if(locked)
			lock(&ifacelock);
		for(sigt = fake[h]; sigt != nil; sigt = (Sigt*)sigt->fun) {
			// don't need to compare indir.
			// same type string but different indir will have
			// different hashes.
			if(mcmp(sigt->name, type->str, type->len) == 0)
			if(sigt->name[type->len] == '\0') {
				if(locked)
					unlock(&ifacelock);
				return sigt;
			}
		}
	}

	sigt = mal(2*sizeof sigt[0]);
	sigt[0].name = mal(type->len + 1);
	mcpy(sigt[0].name, type->str, type->len);
	sigt[0].hash = AFAKE;	// alg
	if(indir)
		sigt[0].offset = 2*sizeof(niliface.data);  // big width
	else
		sigt[0].offset = 1;  // small width
	sigt->fun = (void*)fake[h];
	fake[h] = sigt;
	unlock(&ifacelock);
	return sigt;
}

static int32
cmpstringchars(string a, uint8 *b)
{
	int32 i;
	byte c1, c2;

	for(i=0;; i++) {
		if(i == a->len)
			c1 = 0;
		else
			c1 = a->str[i];
		c2 = b[i];
		if(c1 < c2)
			return -1;
		if(c1 > c2)
			return +1;
		if(c1 == 0)
			return 0;
	}
}

static Sigt*
findtype(string type, bool indir)
{
	int32 i, lo, hi, m;

	lo = 0;
	hi = ngotypesigs;
	while(lo < hi) {
		m = lo + (hi - lo)/2;
		i = cmpstringchars(type, gotypesigs[m]->name);
		if(i == 0)
			return gotypesigs[m];
		if(i < 0)
			hi = m;
		else
			lo = m+1;
	}
	return fakesigt(type, indir);
}


void
sys·Unreflect(uint64 it, string type, bool indir, Iface ret)
{
	Sigt *sigt;

	ret = niliface;

	if(cmpstring(type, emptystring) == 0)
		goto out;

	// if we think the type should be indirect
	// and caller does not, play it safe, return nil.
	sigt = findtype(type, indir);
	if(indir != (sigt[0].offset > sizeof ret.data))
		goto out;

	ret.type = itype(sigi·empty, sigt, 0);
	ret.data = (void*)it;

out:
	FLUSH(&ret);
}


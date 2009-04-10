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
	byte*	name;                   // name of basic type
	Sigt*	link;			// for linking into hash tables
	uint32	thash;                  // hash of type
	uint32	mhash;                  // hash of methods
	uint16	width;			// width of base type in bytes
	uint16	alg;			// algorithm
	// note: on amd64 there is a 32-bit pad here.
	struct {
		byte*	fname;
		uint32	fhash;		// hash of type
		uint32	offset;		// offset of substruct
		void	(*fun)(void);
	} meth[1];			// one or more - last name is nil
};

struct	Sigi
{
	byte*	name;
	uint32	hash;
	uint32	size;			// number of methods
	struct {
		byte*	fname;
		uint32	fhash;
		uint32	perm;		// location of fun in Sigt
	} meth[1];			// [size+1] - last name is nil
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
	prints((int8*)si->name);
	prints(":");
	for(i=0;; i++) {
		name = si->meth[i].fname;
		if(name == nil)
			break;
		prints("[");
		sys·printint(i);
		prints("]\"");
		prints((int8*)name);
		prints("\"");
		sys·printint(si->meth[i].fhash%999);
		prints("/");
		sys·printint(si->meth[i].perm);
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
	prints((int8*)st->name);
	prints(":");
	sys·printint(st->thash%999);	// type hash
	prints(",");
	sys·printint(st->mhash%999);	// method hash
	prints(",");
	sys·printint(st->width);	// width
	prints(",");
	sys·printint(st->alg);	// algorithm
	for(i=0;; i++) {
		name = st->meth[i].fname;
		if(name == nil)
			break;
		prints("[");
		sys·printint(i);
		prints("]\"");
		prints((int8*)name);
		prints("\"");
		sys·printint(st->meth[i].fhash%999);
		prints("/");
		sys·printint(st->meth[i].offset);
		prints("/");
		sys·printpointer(st->meth[i].fun);
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
	if(st) {
		h += st->thash;
		h += st->mhash;
	}

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

	ni = si->size;
	m = malloc(sizeof(*m) + ni*sizeof(m->fun[0]));
	m->sigi = si;
	m->sigt = st;

throw:
	nt = 0;
	for(ni=0;; ni++) {
		iname = si->meth[ni].fname;
		if(iname == nil)
			break;

		// pick up next name from
		// interface signature
		ihash = si->meth[ni].fhash;

		for(;; nt++) {
			// pick up and compare next name
			// from structure signature
			sname = st->meth[nt].fname;
			if(sname == nil) {
				if(!canfail) {
					printf("cannot convert type %s to interface %s: missing method %s\n",
						st->name, si->name, iname);
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
			if(ihash == st->meth[nt].fhash && strcmp(sname, iname) == 0)
				break;
		}
		m->fun[si->meth[ni].perm] = st->meth[nt].fun;
	}
	m->link = hash[h];
	hash[h] = m;
	if(locked)
		unlock(&ifacelock);

	// printf("new itype %p\n", m);
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

	if(iface_debug) {
		prints("T2I sigi=");
		printsigi(si);
		prints(" sigt=");
		printsigt(st);
		prints(" elem=");
		sys·printpointer(*(void**)elem);
		prints("\n");
	}

	wid = st->width;
	alg = st->alg;
	ret = (Iface*)(elem + rnd(wid, sizeof(uintptr)));
	ret->type = itype(si, st, 0);

	if(wid <= sizeof(ret->data))
		algarray[alg].copy(wid, &ret->data, elem);
	else {
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
		prints((int8*)st->name);
		prints("\n");
		throw("interface conversion");
	}

	if(im->sigt != st) {
		prints((int8*)im->sigi->name);
		prints(" is ");
		prints((int8*)im->sigt->name);
		prints(", not ");
		prints((int8*)st->name);
		prints("\n");
		throw("interface conversion");
	}

	alg = st->alg;
	wid = st->width;
	if(wid <= sizeof(i.data))
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


	if(iface_debug) {
		prints("I2T2 sigt=");
		printsigt(st);
		prints(" iface=");
		printiface(i);
		prints("\n");
	}

	ret = (byte*)(&i+1);
	alg = st->alg;
	wid = st->width;
	ok = (bool*)(ret+rnd(wid, 1));

	im = i.type;
	if(im == nil || im->sigt != st) {
		*ok = false;
		sys·memclr(ret, wid);
	} else {
		*ok = true;
		if(wid <= sizeof(i.data))
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
	Sigt *sigt;

	if(a.type == nil)
		return 0;

	sigt = a.type->sigt;
	alg = sigt->alg;
	wid = sigt->width;
	if(algarray[alg].hash == nohash) {
		// calling nohash will throw too,
		// but we can print a better error.
		printf("hash of unhashable type %s\n", sigt->name);
		if(alg == AFAKE)
			throw("fake interface hash");
		throw("interface hash");
	}
	if(wid <= sizeof(a.data))
		return algarray[alg].hash(wid, &a.data);
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

	alg = i1.type->sigt->alg;
	wid = i1.type->sigt->width;

	if(algarray[alg].equal == noequal) {
		// calling noequal will throw too,
		// but we can print a better error.
		printf("comparing uncomparable type %s\n", i1.type->sigt->name);
		if(alg == AFAKE)
			throw("fake interface compare");
		throw("interface compare");
	}

	if(wid <= sizeof(i1.data)) {
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

// ifacethash(i1 any) (ret uint32);
void
sys·ifacethash(Iface i1, uint32 ret)
{
	Itype *im;
	Sigt *st;

	ret = 0;
	im = i1.type;
	if(im != nil) {
		st = im->sigt;
		if(st != nil)
			ret = st->thash;
	}
	FLUSH(&ret);
}

void
sys·printinter(Iface i)
{
	printiface(i);
}

void
sys·Reflect(Iface i, uint64 retit, String rettype, bool retindir)
{
	int32 wid;

	if(i.type == nil) {
		retit = 0;
		rettype = emptystring;
		retindir = false;
	} else {
		retit = (uint64)i.data;
		rettype = gostring(i.type->sigt->name);
		wid = i.type->sigt->width;
		retindir = wid > sizeof(i.data);
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
// for "[]int", so we have to invent one.  The requirements
// on the fake signature are:
//
//	(1) any interface conversion using the signature will fail
//	(2) calling sys.Reflect() returns the args to unreflect
//	(3) the right algorithm type is used, for == and map insertion
//
// (1) is ensured by the fact that we allocate a new Sigt,
// so it will necessarily be != any Sigt in gotypesigs.
// (2) is ensured by storing the type string in the signature
// and setting the width to force the correct value of the bool indir.
// (3) is ensured by sniffing the type string.
//
// Note that (1) is correct behavior: if the program had tested
// for .([]int) instead of .(string) above, then there would be a
// signature with type string "[]int" in gotypesigs, and unreflect
// wouldn't call fakesigt.

static	Sigt*	fake[1009];
static	int32	nfake;

enum
{
	SizeofInt = 4,
	SizeofFloat = 4,
};

// Table of prefixes of names of comparable types.
static	struct {
	int8 *s;
	int8 n;
	int8 alg;
	int8 w;
} cmp[] =
{
	// basic types
	"int", 3+1, AMEM, SizeofInt, // +1 is NUL
	"uint", 4+1, AMEM, SizeofInt,
	"int8", 4+1, AMEM, 1,
	"uint8", 5+1, AMEM, 1,
	"int16", 5+1, AMEM, 2,
	"uint16", 6+1, AMEM, 2,
	"int32", 5+1, AMEM, 4,
	"uint32", 6+1, AMEM, 4,
	"int64", 5+1, AMEM, 8,
	"uint64", 6+1, AMEM, 8,
	"uintptr", 7+1, AMEM, sizeof(uintptr),
	"float", 5+1, AMEM, SizeofFloat,
	"float32", 7+1, AMEM, 4,
	"float64", 7+1, AMEM, 8,
	"bool", 4+1, AMEM, sizeof(bool),

	// string compare is special
	"string", 6+1, ASTRING, sizeof(String),

	// generic types, identified by prefix
	"*", 1, AMEM, sizeof(uintptr),
	"chan ", 5, AMEM, sizeof(uintptr),
	"func(", 5, AMEM, sizeof(uintptr),
	"map[", 4, AMEM, sizeof(uintptr),
};

static Sigt*
fakesigt(String type, bool indir)
{
	Sigt *sigt;
	uint32 h;
	int32 i, locked;

	h = 0;
	for(i=0; i<type.len; i++)
		h = h*37 + type.str[i];
	h += indir;
	h %= nelem(fake);

	for(locked=0; locked<2; locked++) {
		if(locked)
			lock(&ifacelock);
		for(sigt = fake[h]; sigt != nil; sigt = sigt->link) {
			// don't need to compare indir.
			// same type string but different indir will have
			// different hashes.
			if(mcmp(sigt->name, type.str, type.len) == 0)
			if(sigt->name[type.len] == '\0') {
				if(locked)
					unlock(&ifacelock);
				return sigt;
			}
		}
	}

	sigt = malloc(sizeof(*sigt));
	sigt->name = malloc(type.len + 1);
	mcpy(sigt->name, type.str, type.len);

	sigt->alg = AFAKE;
	sigt->width = 1;  // small width
	if(indir)
		sigt->width = 2*sizeof(niliface.data);  // big width

	// AFAKE is like ANOEQ; check whether the type
	// should have a more capable algorithm.
	for(i=0; i<nelem(cmp); i++) {
		if(mcmp((byte*)sigt->name, (byte*)cmp[i].s, cmp[i].n) == 0) {
			sigt->alg = cmp[i].alg;
			sigt->width = cmp[i].w;
			break;
		}
	}

	sigt->link = fake[h];
	fake[h] = sigt;

	unlock(&ifacelock);
	return sigt;
}

static int32
cmpstringchars(String a, uint8 *b)
{
	int32 i;
	byte c1, c2;

	for(i=0;; i++) {
		c1 = 0;
		if(i < a.len)
			c1 = a.str[i];
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
findtype(String type, bool indir)
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
sys·Unreflect(uint64 it, String type, bool indir, Iface ret)
{
	Sigt *sigt;

	ret = niliface;

	if(cmpstring(type, emptystring) == 0)
		goto out;

	// if we think the type should be indirect
	// and caller does not, play it safe, return nil.
	sigt = findtype(type, indir);
	if(indir != (sigt->width > sizeof(ret.data)))
		goto out;

	ret.type = itype(sigi·empty, sigt, 0);
	ret.data = (void*)it;

out:
	FLUSH(&ret);
}


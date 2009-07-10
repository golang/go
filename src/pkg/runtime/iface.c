// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "type.h"

static void
printiface(Iface i)
{
	printf("(%p,%p)", i.tab, i.data);
}

static void
printeface(Eface e)
{
	printf("(%p,%p)", e.type, e.data);
}

/*
 * layout of Itab known to compilers
 */
struct Itab
{
	InterfaceType*	inter;
	Type*	type;
	Itab*	link;
	int32	bad;
	int32	unused;
	void	(*fun[])(void);
};

static	Itab*	hash[1009];
static	Lock	ifacelock;

static Itab*
itab(InterfaceType *inter, Type *type, int32 canfail)
{
	int32 locked;
	int32 ni;
	Method *t, *et;
	IMethod *i, *ei;
	uint32 ihash, h;
	String *iname;
	Itab *m;
	UncommonType *x;

	if(inter->mhdr.nel == 0)
		throw("internal error - misuse of itab");

	// easy case
	x = type->x;
	if(x == nil) {
		if(canfail)
			return nil;
		iname = inter->m[0].name;
		goto throw;
	}

	// compiler has provided some good hash codes for us.
	h = inter->hash;
	h += 17 * type->hash;
	// TODO(rsc): h += 23 * x->mhash ?
	h %= nelem(hash);

	// look twice - once without lock, once with.
	// common case will be no lock contention.
	for(locked=0; locked<2; locked++) {
		if(locked)
			lock(&ifacelock);
		for(m=hash[h]; m!=nil; m=m->link) {
			if(m->inter == inter && m->type == type) {
				if(m->bad) {
					m = nil;
					if(!canfail) {
						// this can only happen if the conversion
						// was already done once using the , ok form
						// and we have a cached negative result.
						// the cached result doesn't record which
						// interface function was missing, so jump
						// down to the interface check, which will
						// do more work but give a better error.
						goto search;
					}
				}
				if(locked)
					unlock(&ifacelock);
				return m;
			}
		}
	}

	ni = inter->mhdr.nel;
	m = malloc(sizeof(*m) + ni*sizeof m->fun[0]);
	m->inter = inter;
	m->type = type;

search:
	// both inter and type have method sorted by hash,
	// so can iterate over both in lock step;
	// the loop is O(ni+nt) not O(ni*nt).
	i = inter->m;
	ei = i + inter->mhdr.nel;
	t = x->m;
	et = t + x->mhdr.nel;
	for(; i < ei; i++) {
		ihash = i->hash;
		iname = i->name;
		for(;; t++) {
			if(t >= et) {
				if(!canfail) {
				throw:
					// didn't find method
					printf("%S is not %S: missing method %S\n",
						*type->string, *inter->string, *iname);
					throw("interface conversion");
					return nil;	// not reached
				}
				m->bad = 1;
				goto out;
			}
			if(t->hash == ihash && t->name == iname)
				break;
		}
		if(m)
			m->fun[i->perm] = t->ifn;
	}

out:
	m->link = hash[h];
	hash[h] = m;
	if(locked)
		unlock(&ifacelock);
	if(m->bad)
		return nil;
	return m;
}

static void
copyin(Type *t, void *src, void **dst)
{
	int32 wid, alg;
	void *p;

	wid = t->size;
	alg = t->alg;

	if(wid <= sizeof(*dst))
		algarray[alg].copy(wid, dst, src);
	else {
		p = mal(wid);
		algarray[alg].copy(wid, p, src);
		*dst = p;
	}
}

static void
copyout(Type *t, void **src, void *dst)
{
	int32 wid, alg;

	wid = t->size;
	alg = t->alg;

	if(wid <= sizeof(*src))
		algarray[alg].copy(wid, dst, src);
	else
		algarray[alg].copy(wid, dst, *src);
}

// ifaceT2I(sigi *byte, sigt *byte, elem any) (ret Iface);
#pragma textflag 7
void
sys·ifaceT2I(InterfaceType *inter, Type *t, ...)
{
	byte *elem;
	Iface *ret;
	int32 wid;

	elem = (byte*)(&t+1);
	wid = t->size;
	ret = (Iface*)(elem + rnd(wid, Structrnd));
	ret->tab = itab(inter, t, 0);
	copyin(t, elem, &ret->data);
}

// ifaceT2E(sigt *byte, elem any) (ret Eface);
#pragma textflag 7
void
sys·ifaceT2E(Type *t, ...)
{
	byte *elem;
	Eface *ret;
	int32 wid;

	elem = (byte*)(&t+1);
	wid = t->size;
	ret = (Eface*)(elem + rnd(wid, Structrnd));

	ret->type = t;
	copyin(t, elem, &ret->data);
}

// ifaceI2T(sigt *byte, iface any) (ret any);
#pragma textflag 7
void
sys·ifaceI2T(Type *t, Iface i, ...)
{
	Itab *tab;
	byte *ret;

	ret = (byte*)(&i+1);
	tab = i.tab;
	if(tab == nil) {
		printf("interface is nil, not %S\n", *t->string);
		throw("interface conversion");
	}
	if(tab->type != t) {
		printf("%S is %S, not %S\n", *tab->inter->string, *tab->type->string, *t->string);
		throw("interface conversion");
	}
	copyout(t, &i.data, ret);
}

// ifaceI2T2(sigt *byte, i Iface) (ret any, ok bool);
#pragma textflag 7
void
sys·ifaceI2T2(Type *t, Iface i, ...)
{
	byte *ret;
	bool *ok;
	int32 wid;

	ret = (byte*)(&i+1);
	wid = t->size;
	ok = (bool*)(ret+rnd(wid, 1));

	if(i.tab == nil || i.tab->type != t) {
		*ok = false;
		sys·memclr(ret, wid);
		return;
	}

	*ok = true;
	copyout(t, &i.data, ret);
}

// ifaceE2T(sigt *byte, e Eface) (ret any);
#pragma textflag 7
void
sys·ifaceE2T(Type *t, Eface e, ...)
{
	byte *ret;

	ret = (byte*)(&e+1);

	if(e.type != t) {
		if(e.type == nil)
			printf("interface is nil, not %S\n", *t->string);
		else
			printf("interface is %S, not %S\n", *e.type->string, *t->string);
		throw("interface conversion");
	}
	copyout(t, &e.data, ret);
}

// ifaceE2T2(sigt *byte, iface any) (ret any, ok bool);
#pragma textflag 7
void
sys·ifaceE2T2(Type *t, Eface e, ...)
{
	byte *ret;
	bool *ok;
	int32 wid;

	ret = (byte*)(&e+1);
	wid = t->size;
	ok = (bool*)(ret+rnd(wid, 1));

	if(t != e.type) {
		*ok = false;
		sys·memclr(ret, wid);
		return;
	}

	*ok = true;
	copyout(t, &e.data, ret);
}

// ifaceI2E(sigi *byte, iface any) (ret any);
// TODO(rsc): Move to back end, throw away function.
void
sys·ifaceI2E(Iface i, Eface ret)
{
	Itab *tab;

	ret.data = i.data;
	tab = i.tab;
	if(tab == nil)
		ret.type = nil;
	else
		ret.type = tab->type;
	FLUSH(&ret);
}

// ifaceI2I(sigi *byte, iface any) (ret any);
// called only for implicit (no type assertion) conversions.
// converting nil is okay.
void
sys·ifaceI2I(InterfaceType *inter, Iface i, Iface ret)
{
	Itab *tab;

	tab = i.tab;
	if(tab == nil) {
		// If incoming interface is uninitialized (zeroed)
		// make the outgoing interface zeroed as well.
		ret.tab = nil;
		ret.data = nil;
	} else {
		ret = i;
		if(tab->inter != inter)
			ret.tab = itab(inter, tab->type, 0);
	}

	FLUSH(&ret);
}

// ifaceI2Ix(sigi *byte, iface any) (ret any);
// called only for explicit conversions (with type assertion).
// converting nil is not okay.
void
sys·ifaceI2Ix(InterfaceType *inter, Iface i, Iface ret)
{
	Itab *tab;

	tab = i.tab;
	if(tab == nil) {
		// explicit conversions require non-nil interface value.
		printf("interface is nil, not %S\n", *inter->string);
		throw("interface conversion");
	} else {
		ret = i;
		if(tab->inter != inter)
			ret.tab = itab(inter, tab->type, 0);
	}

	FLUSH(&ret);
}

// ifaceI2I2(sigi *byte, iface any) (ret any, ok bool);
void
sys·ifaceI2I2(InterfaceType *inter, Iface i, Iface ret, bool ok)
{
	Itab *tab;

	tab = i.tab;
	if(tab == nil) {
		// If incoming interface is nil, the conversion fails.
		ret.tab = nil;
		ret.data = nil;
		ok = false;
	} else {
		ret = i;
		ok = true;
		if(tab->inter != inter) {
			ret.tab = itab(inter, tab->type, 1);
			if(ret.tab == nil) {
				ret.data = nil;
				ok = false;
			}
		}
	}

	FLUSH(&ret);
	FLUSH(&ok);
}

// ifaceE2I(sigi *byte, iface any) (ret any);
// Called only for explicit conversions (with type assertion).
void
ifaceE2I(InterfaceType *inter, Eface e, Iface *ret)
{
	Type *t;

	t = e.type;
	if(t == nil) {
		// explicit conversions require non-nil interface value.
		printf("interface is nil, not %S\n", *inter->string);
		throw("interface conversion");
	} else {
		ret->data = e.data;
		ret->tab = itab(inter, t, 0);
	}
}

// ifaceE2I(sigi *byte, iface any) (ret any);
// Called only for explicit conversions (with type assertion).
void
sys·ifaceE2I(InterfaceType *inter, Eface e, Iface ret)
{
	ifaceE2I(inter, e, &ret);
}

// ifaceE2I2(sigi *byte, iface any) (ret any, ok bool);
void
sys·ifaceE2I2(InterfaceType *inter, Eface e, Iface ret, bool ok)
{
	Type *t;

	t = e.type;
	ok = true;
	if(t == nil) {
		// If incoming interface is nil, the conversion fails.
		ret.data = nil;
		ret.tab = nil;
		ok = false;
	} else {
		ret.data = e.data;
		ret.tab = itab(inter, t, 1);
		if(ret.tab == nil) {
			ret.data = nil;
			ok = false;
		}
	}
	FLUSH(&ret);
	FLUSH(&ok);
}

static uintptr
ifacehash1(void *data, Type *t)
{
	int32 alg, wid;

	if(t == nil)
		return 0;

	alg = t->alg;
	wid = t->size;
	if(algarray[alg].hash == nohash) {
		// calling nohash will throw too,
		// but we can print a better error.
		printf("hash of unhashable type %S\n", *t->string);
		if(alg == AFAKE)
			throw("fake interface hash");
		throw("interface hash");
	}
	if(wid <= sizeof(data))
		return algarray[alg].hash(wid, &data);
	return algarray[alg].hash(wid, data);
}

uintptr
ifacehash(Iface a)
{
	if(a.tab == nil)
		return 0;
	return ifacehash1(a.data, a.tab->type);
}

uintptr
efacehash(Eface a)
{
	return ifacehash1(a.data, a.type);
}

static bool
ifaceeq1(void *data1, void *data2, Type *t)
{
	int32 alg, wid;

	alg = t->alg;
	wid = t->size;

	if(algarray[alg].equal == noequal) {
		// calling noequal will throw too,
		// but we can print a better error.
		printf("comparing uncomparable type %S\n", *t->string);
		if(alg == AFAKE)
			throw("fake interface compare");
		throw("interface compare");
	}

	if(wid <= sizeof(data1))
		return algarray[alg].equal(wid, &data1, &data2);
	return algarray[alg].equal(wid, data1, data2);
}

bool
ifaceeq(Iface i1, Iface i2)
{
	if(i1.tab != i2.tab)
		return false;
	if(i1.tab == nil)
		return true;
	return ifaceeq1(i1.data, i2.data, i1.tab->type);
}

bool
efaceeq(Eface e1, Eface e2)
{
	if(e1.type != e2.type)
		return false;
	if(e1.type == nil)
		return true;
	return ifaceeq1(e1.data, e2.data, e1.type);
}

// ifaceeq(i1 any, i2 any) (ret bool);
void
sys·ifaceeq(Iface i1, Iface i2, bool ret)
{
	ret = ifaceeq(i1, i2);
	FLUSH(&ret);
}

// efaceeq(i1 any, i2 any) (ret bool)
void
sys·efaceeq(Eface e1, Eface e2, bool ret)
{
	ret = efaceeq(e1, e2);
	FLUSH(&ret);
}

// ifacethash(i1 any) (ret uint32);
void
sys·ifacethash(Iface i1, uint32 ret)
{
	Itab *tab;

	ret = 0;
	tab = i1.tab;
	if(tab != nil)
		ret = tab->type->hash;
	FLUSH(&ret);
}

// efacethash(e1 any) (ret uint32)
void
sys·efacethash(Eface e1, uint32 ret)
{
	Type *t;

	ret = 0;
	t = e1.type;
	if(t != nil)
		ret = t->hash;
	FLUSH(&ret);
}

void
sys·printiface(Iface i)
{
	printiface(i);
}

void
sys·printeface(Eface e)
{
	printeface(e);
}

void
unsafe·Typeof(Eface e, Eface ret)
{
	if(e.type == nil) {
		ret.type = nil;
		ret.data = nil;
	} else
		ret = *(Eface*)e.type;
	FLUSH(&ret);
}

void
unsafe·Reflect(Eface e, Eface rettype, void *retaddr)
{
	uintptr *p;
	uintptr x;

	if(e.type == nil) {
		rettype.type = nil;
		rettype.data = nil;
		retaddr = 0;
	} else {
		rettype = *(Eface*)e.type;
		if(e.type->size <= sizeof(uintptr)) {
			// Copy data into x ...
			x = 0;
			algarray[e.type->alg].copy(e.type->size, &x, &e.data);

			// but then build pointer to x so that Reflect
			// always returns pointer to data.
			p = mallocgc(sizeof(uintptr));
			*p = x;
		} else {
			// Already a pointer, but still make a copy,
			// to preserve value semantics for interface data.
			p = mallocgc(e.type->size);
			algarray[e.type->alg].copy(e.type->size, p, e.data);
		}
		retaddr = p;
	}
	FLUSH(&rettype);
	FLUSH(&retaddr);
}

void
unsafe·Unreflect(Iface typ, void *addr, Eface e)
{
	// Reflect library has reinterpreted typ
	// as its own kind of type structure.
	// We know that the pointer to the original
	// type structure sits before the data pointer.
	e.type = (Type*)((Eface*)typ.data-1);

	// Interface holds either pointer to data
	// or copy of original data.
	if(e.type->size <= sizeof(uintptr))
		algarray[e.type->alg].copy(e.type->size, &e.data, addr);
	else {
		// Easier: already a pointer to data.
		// TODO(rsc): Should this make a copy?
		e.data = addr;
	}

	FLUSH(&e);
}

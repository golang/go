// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "type.h"
#include "malloc.h"

void
runtime·printiface(Iface i)
{
	runtime·printf("(%p,%p)", i.tab, i.data);
}

void
runtime·printeface(Eface e)
{
	runtime·printf("(%p,%p)", e.type, e.data);
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
	uint32 h;
	String *iname;
	Itab *m;
	UncommonType *x;
	Type *itype;
	Eface err;

	if(inter->mhdr.len == 0)
		runtime·throw("internal error - misuse of itab");

	locked = 0;

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
			runtime·lock(&ifacelock);
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
					runtime·unlock(&ifacelock);
				return m;
			}
		}
	}

	ni = inter->mhdr.len;
	m = runtime·malloc(sizeof(*m) + ni*sizeof m->fun[0]);
	m->inter = inter;
	m->type = type;

search:
	// both inter and type have method sorted by name,
	// and interface names are unique,
	// so can iterate over both in lock step;
	// the loop is O(ni+nt) not O(ni*nt).
	i = inter->m;
	ei = i + inter->mhdr.len;
	t = x->m;
	et = t + x->mhdr.len;
	for(; i < ei; i++) {
		itype = i->type;
		iname = i->name;
		for(;; t++) {
			if(t >= et) {
				if(!canfail) {
				throw:
					// didn't find method
					runtime·newTypeAssertionError(nil, type, inter,
						nil, type->string, inter->string,
						iname, &err);
					if(locked)
						runtime·unlock(&ifacelock);
					runtime·panic(err);
					return nil;	// not reached
				}
				m->bad = 1;
				goto out;
			}
			if(t->mtyp == itype && t->name == iname)
				break;
		}
		if(m)
			m->fun[i - inter->m] = t->ifn;
	}

out:
	m->link = hash[h];
	hash[h] = m;
	if(locked)
		runtime·unlock(&ifacelock);
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
		runtime·algarray[alg].copy(wid, dst, src);
	else {
		p = runtime·mal(wid);
		runtime·algarray[alg].copy(wid, p, src);
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
		runtime·algarray[alg].copy(wid, dst, src);
	else
		runtime·algarray[alg].copy(wid, dst, *src);
}

// func convT2I(typ *byte, typ2 *byte, elem any) (ret any)
#pragma textflag 7
void
runtime·convT2I(Type *t, InterfaceType *inter, ...)
{
	byte *elem;
	Iface *ret;
	int32 wid;

	elem = (byte*)(&inter+1);
	wid = t->size;
	ret = (Iface*)(elem + runtime·rnd(wid, Structrnd));
	ret->tab = itab(inter, t, 0);
	copyin(t, elem, &ret->data);
}

// func convT2E(typ *byte, elem any) (ret any)
#pragma textflag 7
void
runtime·convT2E(Type *t, ...)
{
	byte *elem;
	Eface *ret;
	int32 wid;

	elem = (byte*)(&t+1);
	wid = t->size;
	ret = (Eface*)(elem + runtime·rnd(wid, Structrnd));
	ret->type = t;
	copyin(t, elem, &ret->data);
}

// func ifaceI2T(typ *byte, iface any) (ret any)
#pragma textflag 7
void
runtime·assertI2T(Type *t, Iface i, ...)
{
	Itab *tab;
	byte *ret;
	Eface err;

	ret = (byte*)(&i+1);
	tab = i.tab;
	if(tab == nil) {
		runtime·newTypeAssertionError(nil, nil, t,
			nil, nil, t->string,
			nil, &err);
		runtime·panic(err);
	}
	if(tab->type != t) {
		runtime·newTypeAssertionError(tab->inter, tab->type, t,
			tab->inter->string, tab->type->string, t->string,
			nil, &err);
		runtime·panic(err);
	}
	copyout(t, &i.data, ret);
}

// func ifaceI2T2(typ *byte, iface any) (ret any, ok bool)
#pragma textflag 7
void
runtime·assertI2T2(Type *t, Iface i, ...)
{
	byte *ret;
	bool *ok;
	int32 wid;

	ret = (byte*)(&i+1);
	wid = t->size;
	ok = (bool*)(ret+runtime·rnd(wid, 1));

	if(i.tab == nil || i.tab->type != t) {
		*ok = false;
		runtime·memclr(ret, wid);
		return;
	}

	*ok = true;
	copyout(t, &i.data, ret);
}

// func ifaceE2T(typ *byte, iface any) (ret any)
#pragma textflag 7
void
runtime·assertE2T(Type *t, Eface e, ...)
{
	byte *ret;
	Eface err;

	ret = (byte*)(&e+1);

	if(e.type == nil) {
		runtime·newTypeAssertionError(nil, nil, t,
			nil, nil, t->string,
			nil, &err);
		runtime·panic(err);
	}
	if(e.type != t) {
		runtime·newTypeAssertionError(nil, e.type, t,
			nil, e.type->string, t->string,
			nil, &err);
		runtime·panic(err);
	}
	copyout(t, &e.data, ret);
}

// func ifaceE2T2(sigt *byte, iface any) (ret any, ok bool);
#pragma textflag 7
void
runtime·assertE2T2(Type *t, Eface e, ...)
{
	byte *ret;
	bool *ok;
	int32 wid;

	ret = (byte*)(&e+1);
	wid = t->size;
	ok = (bool*)(ret+runtime·rnd(wid, 1));

	if(t != e.type) {
		*ok = false;
		runtime·memclr(ret, wid);
		return;
	}

	*ok = true;
	copyout(t, &e.data, ret);
}

// func convI2E(elem any) (ret any)
#pragma textflag 7
void
runtime·convI2E(Iface i, Eface ret)
{
	Itab *tab;

	ret.data = i.data;
	if((tab = i.tab) == nil)
		ret.type = nil;
	else
		ret.type = tab->type;
	FLUSH(&ret);
}

// func ifaceI2E(typ *byte, iface any) (ret any)
#pragma textflag 7
void
runtime·assertI2E(InterfaceType* inter, Iface i, Eface ret)
{
	Itab *tab;
	Eface err;

	tab = i.tab;
	if(tab == nil) {
		// explicit conversions require non-nil interface value.
		runtime·newTypeAssertionError(nil, nil, inter,
			nil, nil, inter->string,
			nil, &err);
		runtime·panic(err);
	}
	ret.data = i.data;
	ret.type = tab->type;
	FLUSH(&ret);
}

// func ifaceI2E2(typ *byte, iface any) (ret any, ok bool)
#pragma textflag 7
void
runtime·assertI2E2(InterfaceType* inter, Iface i, Eface ret, bool ok)
{
	Itab *tab;

	USED(inter);
	tab = i.tab;
	if(tab == nil) {
		ret.type = nil;
		ok = 0;
	} else {
		ret.type = tab->type;
		ok = 1;
	}
	ret.data = i.data;
	FLUSH(&ret);
	FLUSH(&ok);
}

// func convI2I(typ *byte, elem any) (ret any)
#pragma textflag 7
void
runtime·convI2I(InterfaceType* inter, Iface i, Iface ret)
{
	Itab *tab;
	
	ret.data = i.data;
	if((tab = i.tab) == nil)
		ret.tab = nil;
	else if(tab->inter == inter)
		ret.tab = tab;
	else
		ret.tab = itab(inter, tab->type, 0);
	FLUSH(&ret);
}

void
runtime·ifaceI2I(InterfaceType *inter, Iface i, Iface *ret)
{
	Itab *tab;
	Eface err;

	tab = i.tab;
	if(tab == nil) {
		// explicit conversions require non-nil interface value.
		runtime·newTypeAssertionError(nil, nil, inter,
			nil, nil, inter->string,
			nil, &err);
		runtime·panic(err);
	}
	ret->data = i.data;
	ret->tab = itab(inter, tab->type, 0);
}

// func ifaceI2I(sigi *byte, iface any) (ret any)
#pragma textflag 7
void
runtime·assertI2I(InterfaceType* inter, Iface i, Iface ret)
{
	runtime·ifaceI2I(inter, i, &ret);
}

// func ifaceI2I2(sigi *byte, iface any) (ret any, ok bool)
#pragma textflag 7
void
runtime·assertI2I2(InterfaceType *inter, Iface i, Iface ret, bool ok)
{
	Itab *tab;

	tab = i.tab;
	if(tab != nil && (tab->inter == inter || (tab = itab(inter, tab->type, 1)) != nil)) {
		ret.data = i.data;
		ret.tab = tab;
		ok = 1;
	} else {
		ret.data = 0;
		ret.tab = 0;
		ok = 0;
	}
	FLUSH(&ret);
	FLUSH(&ok);
}

void
runtime·ifaceE2I(InterfaceType *inter, Eface e, Iface *ret)
{
	Type *t;
	Eface err;

	t = e.type;
	if(t == nil) {
		// explicit conversions require non-nil interface value.
		runtime·newTypeAssertionError(nil, nil, inter,
			nil, nil, inter->string,
			nil, &err);
		runtime·panic(err);
	}
	ret->data = e.data;
	ret->tab = itab(inter, t, 0);
}

// func ifaceE2I(sigi *byte, iface any) (ret any)
#pragma textflag 7
void
runtime·assertE2I(InterfaceType* inter, Eface e, Iface ret)
{
	runtime·ifaceE2I(inter, e, &ret);
}

// ifaceE2I2(sigi *byte, iface any) (ret any, ok bool)
#pragma textflag 7
void
runtime·assertE2I2(InterfaceType *inter, Eface e, Iface ret, bool ok)
{
	if(e.type == nil) {
		ok = 0;
		ret.data = nil;
		ret.tab = nil;
	} else if((ret.tab = itab(inter, e.type, 1)) == nil) {
		ok = 0;
		ret.data = nil;
	} else {
		ok = 1;
		ret.data = e.data;
	}
	FLUSH(&ret);
	FLUSH(&ok);
}

// func ifaceE2E(typ *byte, iface any) (ret any)
#pragma textflag 7
void
runtime·assertE2E(InterfaceType* inter, Eface e, Eface ret)
{
	Type *t;
	Eface err;

	t = e.type;
	if(t == nil) {
		// explicit conversions require non-nil interface value.
		runtime·newTypeAssertionError(nil, nil, inter,
			nil, nil, inter->string,
			nil, &err);
		runtime·panic(err);
	}
	ret = e;
	FLUSH(&ret);
}

// func ifaceE2E2(iface any) (ret any, ok bool)
#pragma textflag 7
void
runtime·assertE2E2(InterfaceType* inter, Eface e, Eface ret, bool ok)
{
	USED(inter);
	ret = e;
	ok = e.type != nil;
	FLUSH(&ret);
	FLUSH(&ok);
}

static uintptr
ifacehash1(void *data, Type *t)
{
	int32 alg, wid;
	Eface err;

	if(t == nil)
		return 0;

	alg = t->alg;
	wid = t->size;
	if(runtime·algarray[alg].hash == runtime·nohash) {
		// calling nohash will panic too,
		// but we can print a better error.
		runtime·newErrorString(runtime·catstring(runtime·gostringnocopy((byte*)"hash of unhashable type "), *t->string), &err);
		runtime·panic(err);
	}
	if(wid <= sizeof(data))
		return runtime·algarray[alg].hash(wid, &data);
	return runtime·algarray[alg].hash(wid, data);
}

uintptr
runtime·ifacehash(Iface a)
{
	if(a.tab == nil)
		return 0;
	return ifacehash1(a.data, a.tab->type);
}

uintptr
runtime·efacehash(Eface a)
{
	return ifacehash1(a.data, a.type);
}

static bool
ifaceeq1(void *data1, void *data2, Type *t)
{
	int32 alg, wid;
	Eface err;

	alg = t->alg;
	wid = t->size;

	if(runtime·algarray[alg].equal == runtime·noequal) {
		// calling noequal will panic too,
		// but we can print a better error.
		runtime·newErrorString(runtime·catstring(runtime·gostringnocopy((byte*)"comparing uncomparable type "), *t->string), &err);
		runtime·panic(err);
	}

	if(wid <= sizeof(data1))
		return runtime·algarray[alg].equal(wid, &data1, &data2);
	return runtime·algarray[alg].equal(wid, data1, data2);
}

bool
runtime·ifaceeq_c(Iface i1, Iface i2)
{
	if(i1.tab != i2.tab)
		return false;
	if(i1.tab == nil)
		return true;
	return ifaceeq1(i1.data, i2.data, i1.tab->type);
}

bool
runtime·efaceeq_c(Eface e1, Eface e2)
{
	if(e1.type != e2.type)
		return false;
	if(e1.type == nil)
		return true;
	return ifaceeq1(e1.data, e2.data, e1.type);
}

// ifaceeq(i1 any, i2 any) (ret bool);
void
runtime·ifaceeq(Iface i1, Iface i2, bool ret)
{
	ret = runtime·ifaceeq_c(i1, i2);
	FLUSH(&ret);
}

// efaceeq(i1 any, i2 any) (ret bool)
void
runtime·efaceeq(Eface e1, Eface e2, bool ret)
{
	ret = runtime·efaceeq_c(e1, e2);
	FLUSH(&ret);
}

// ifacethash(i1 any) (ret uint32);
void
runtime·ifacethash(Iface i1, uint32 ret)
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
runtime·efacethash(Eface e1, uint32 ret)
{
	Type *t;

	ret = 0;
	t = e1.type;
	if(t != nil)
		ret = t->hash;
	FLUSH(&ret);
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
			runtime·algarray[e.type->alg].copy(e.type->size, &x, &e.data);

			// but then build pointer to x so that Reflect
			// always returns pointer to data.
			p = runtime·mal(sizeof(uintptr));
			*p = x;
		} else {
			// Already a pointer, but still make a copy,
			// to preserve value semantics for interface data.
			p = runtime·mal(e.type->size);
			runtime·algarray[e.type->alg].copy(e.type->size, p, e.data);
		}
		retaddr = p;
	}
	FLUSH(&rettype);
	FLUSH(&retaddr);
}

void
unsafe·Unreflect(Eface typ, void *addr, Eface e)
{
	// Reflect library has reinterpreted typ
	// as its own kind of type structure.
	// We know that the pointer to the original
	// type structure sits before the data pointer.
	e.type = (Type*)((Eface*)typ.data-1);

	// Interface holds either pointer to data
	// or copy of original data.
	if(e.type->size <= sizeof(uintptr))
		runtime·algarray[e.type->alg].copy(e.type->size, &e.data, addr);
	else {
		// Easier: already a pointer to data.
		// TODO(rsc): Should this make a copy?
		e.data = addr;
	}

	FLUSH(&e);
}

void
unsafe·New(Eface typ, void *ret)
{
	Type *t;

	// Reflect library has reinterpreted typ
	// as its own kind of type structure.
	// We know that the pointer to the original
	// type structure sits before the data pointer.
	t = (Type*)((Eface*)typ.data-1);

	if(t->kind&KindNoPointers)
		ret = runtime·mallocgc(t->size, FlagNoPointers, 1, 1);
	else
		ret = runtime·mal(t->size);
	FLUSH(&ret);
}

void
unsafe·NewArray(Eface typ, uint32 n, void *ret)
{
	uint64 size;
	Type *t;

	// Reflect library has reinterpreted typ
	// as its own kind of type structure.
	// We know that the pointer to the original
	// type structure sits before the data pointer.
	t = (Type*)((Eface*)typ.data-1);
	
	size = n*t->size;
	if(t->kind&KindNoPointers)
		ret = runtime·mallocgc(size, FlagNoPointers, 1, 1);
	else
		ret = runtime·mal(size);
	FLUSH(&ret);
}

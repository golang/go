// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"
#include "type.h"
#include "typekind.h"
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
	String *iname, *ipkgPath;
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
		for(m=runtime·atomicloadp(&hash[h]); m!=nil; m=m->link) {
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
		ipkgPath = i->pkgPath;
		for(;; t++) {
			if(t >= et) {
				if(!canfail) {
				throw:
					// didn't find method
					runtime·newTypeAssertionError(
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
			if(t->mtyp == itype && t->name == iname && t->pkgPath == ipkgPath)
				break;
		}
		if(m)
			m->fun[i - inter->m] = t->ifn;
	}

out:
	if(!locked)
		runtime·panicstring("invalid itab locking");
	m->link = hash[h];
	runtime·atomicstorep(&hash[h], m);
	runtime·unlock(&ifacelock);
	if(m->bad)
		return nil;
	return m;
}

static void
copyin(Type *t, void *src, void **dst)
{
	uintptr size;
	void *p;
	Alg *alg;

	size = t->size;
	alg = t->alg;

	if(size <= sizeof(*dst))
		alg->copy(size, dst, src);
	else {
		p = runtime·mal(size);
		alg->copy(size, p, src);
		*dst = p;
	}
}

static void
copyout(Type *t, void **src, void *dst)
{
	uintptr size;
	Alg *alg;

	size = t->size;
	alg = t->alg;

	if(size <= sizeof(*src))
		alg->copy(size, dst, src);
	else
		alg->copy(size, dst, *src);
}

#pragma textflag 7
void
runtime·typ2Itab(Type *t, InterfaceType *inter, Itab **cache, Itab *ret)
{
	Itab *tab;

	tab = itab(inter, t, 0);
	runtime·atomicstorep(cache, tab);
	ret = tab;
	FLUSH(&ret);
}

// func convT2I(typ *byte, typ2 *byte, cache **byte, elem any) (ret any)
#pragma textflag 7
void
runtime·convT2I(Type *t, InterfaceType *inter, Itab **cache, ...)
{
	byte *elem;
	Iface *ret;
	Itab *tab;
	int32 wid;

	elem = (byte*)(&cache+1);
	wid = t->size;
	ret = (Iface*)(elem + ROUND(wid, Structrnd));
	tab = runtime·atomicloadp(cache);
	if(!tab) {
		tab = itab(inter, t, 0);
		runtime·atomicstorep(cache, tab);
	}
	ret->tab = tab;
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
	ret = (Eface*)(elem + ROUND(wid, Structrnd));
	ret->type = t;
	copyin(t, elem, &ret->data);
}

static void assertI2Tret(Type *t, Iface i, byte *ret);

// func ifaceI2T(typ *byte, iface any) (ret any)
#pragma textflag 7
void
runtime·assertI2T(Type *t, Iface i, ...)
{
	byte *ret;

	ret = (byte*)(&i+1);
	assertI2Tret(t, i, ret);
}

static void
assertI2Tret(Type *t, Iface i, byte *ret)
{
	Itab *tab;
	Eface err;

	tab = i.tab;
	if(tab == nil) {
		runtime·newTypeAssertionError(
			nil, nil, t->string,
			nil, &err);
		runtime·panic(err);
	}
	if(tab->type != t) {
		runtime·newTypeAssertionError(
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
	ok = (bool*)(ret + wid);

	if(i.tab == nil || i.tab->type != t) {
		*ok = false;
		runtime·memclr(ret, wid);
		return;
	}

	*ok = true;
	copyout(t, &i.data, ret);
}

void
runtime·assertI2TOK(Type *t, Iface i, bool ok)
{
	ok = i.tab!=nil && i.tab->type==t;
	FLUSH(&ok);
}

static void assertE2Tret(Type *t, Eface e, byte *ret);

// func ifaceE2T(typ *byte, iface any) (ret any)
#pragma textflag 7
void
runtime·assertE2T(Type *t, Eface e, ...)
{
	byte *ret;

	ret = (byte*)(&e+1);
	assertE2Tret(t, e, ret);
}

static void
assertE2Tret(Type *t, Eface e, byte *ret)
{
	Eface err;

	if(e.type == nil) {
		runtime·newTypeAssertionError(
			nil, nil, t->string,
			nil, &err);
		runtime·panic(err);
	}
	if(e.type != t) {
		runtime·newTypeAssertionError(
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
	ok = (bool*)(ret + wid);

	if(t != e.type) {
		*ok = false;
		runtime·memclr(ret, wid);
		return;
	}

	*ok = true;
	copyout(t, &e.data, ret);
}

void
runtime·assertE2TOK(Type *t, Eface e, bool ok)
{
	ok = t==e.type;
	FLUSH(&ok);
}

// func convI2E(elem any) (ret any)
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
void
runtime·assertI2E(InterfaceType* inter, Iface i, Eface ret)
{
	Itab *tab;
	Eface err;

	tab = i.tab;
	if(tab == nil) {
		// explicit conversions require non-nil interface value.
		runtime·newTypeAssertionError(
			nil, nil, inter->string,
			nil, &err);
		runtime·panic(err);
	}
	ret.data = i.data;
	ret.type = tab->type;
	FLUSH(&ret);
}

// func ifaceI2E2(typ *byte, iface any) (ret any, ok bool)
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
		runtime·newTypeAssertionError(
			nil, nil, inter->string,
			nil, &err);
		runtime·panic(err);
	}
	ret->data = i.data;
	ret->tab = itab(inter, tab->type, 0);
}

// func ifaceI2I(sigi *byte, iface any) (ret any)
void
runtime·assertI2I(InterfaceType* inter, Iface i, Iface ret)
{
	runtime·ifaceI2I(inter, i, &ret);
}

// func ifaceI2I2(sigi *byte, iface any) (ret any, ok bool)
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
		runtime·newTypeAssertionError(
			nil, nil, inter->string,
			nil, &err);
		runtime·panic(err);
	}
	ret->data = e.data;
	ret->tab = itab(inter, t, 0);
}

// For reflect
//	func ifaceE2I(t *InterfaceType, e interface{}, dst *Iface)
void
reflect·ifaceE2I(InterfaceType *inter, Eface e, Iface *dst)
{
	runtime·ifaceE2I(inter, e, dst);
}

// func ifaceE2I(sigi *byte, iface any) (ret any)
void
runtime·assertE2I(InterfaceType* inter, Eface e, Iface ret)
{
	runtime·ifaceE2I(inter, e, &ret);
}

// ifaceE2I2(sigi *byte, iface any) (ret any, ok bool)
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
void
runtime·assertE2E(InterfaceType* inter, Eface e, Eface ret)
{
	Type *t;
	Eface err;

	t = e.type;
	if(t == nil) {
		// explicit conversions require non-nil interface value.
		runtime·newTypeAssertionError(
			nil, nil, inter->string,
			nil, &err);
		runtime·panic(err);
	}
	ret = e;
	FLUSH(&ret);
}

// func ifaceE2E2(iface any) (ret any, ok bool)
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
ifacehash1(void *data, Type *t, uintptr h)
{
	Alg *alg;
	uintptr size;
	Eface err;

	if(t == nil)
		return 0;

	alg = t->alg;
	size = t->size;
	if(alg->hash == runtime·nohash) {
		// calling nohash will panic too,
		// but we can print a better error.
		runtime·newErrorString(runtime·catstring(runtime·gostringnocopy((byte*)"hash of unhashable type "), *t->string), &err);
		runtime·panic(err);
	}
	if(size <= sizeof(data))
		alg->hash(&h, size, &data);
	else
		alg->hash(&h, size, data);
	return h;
}

uintptr
runtime·ifacehash(Iface a, uintptr h)
{
	if(a.tab == nil)
		return h;
	return ifacehash1(a.data, a.tab->type, h);
}

uintptr
runtime·efacehash(Eface a, uintptr h)
{
	return ifacehash1(a.data, a.type, h);
}

static bool
ifaceeq1(void *data1, void *data2, Type *t)
{
	uintptr size;
	Alg *alg;
	Eface err;
	bool eq;

	alg = t->alg;
	size = t->size;

	if(alg->equal == runtime·noequal) {
		// calling noequal will panic too,
		// but we can print a better error.
		runtime·newErrorString(runtime·catstring(runtime·gostringnocopy((byte*)"comparing uncomparable type "), *t->string), &err);
		runtime·panic(err);
	}

	eq = 0;
	if(size <= sizeof(data1))
		alg->equal(&eq, size, &data1, &data2);
	else
		alg->equal(&eq, size, data1, data2);
	return eq;
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
reflect·unsafe_Typeof(Eface e, Eface ret)
{
	if(e.type == nil) {
		ret.type = nil;
		ret.data = nil;
	} else {
		ret = *(Eface*)(e.type);
	}
	FLUSH(&ret);
}

void
reflect·unsafe_New(Type *t, void *ret)
{
	uint32 flag;

	flag = t->kind&KindNoPointers ? FlagNoPointers : 0;
	ret = runtime·mallocgc(t->size, flag, 1, 1);

	if(UseSpanType && !flag) {
		if(false) {
			runtime·printf("unsafe_New %S: %p\n", *t->string, ret);
		}
		runtime·settype(ret, (uintptr)t | TypeInfo_SingleObject);
	}

	FLUSH(&ret);
}

void
reflect·unsafe_NewArray(Type *t, intgo n, void *ret)
{
	uint64 size;

	size = n*t->size;
	if(size == 0)
		ret = (byte*)&runtime·zerobase;
	else if(t->kind&KindNoPointers)
		ret = runtime·mallocgc(size, FlagNoPointers, 1, 1);
	else {
		ret = runtime·mallocgc(size, 0, 1, 1);

		if(UseSpanType) {
			if(false) {
				runtime·printf("unsafe_NewArray [%D]%S: %p\n", (int64)n, *t->string, ret);
			}
			runtime·settype(ret, (uintptr)t | TypeInfo_Array);
		}
	}

	FLUSH(&ret);
}

void
reflect·typelinks(Slice ret)
{
	extern Type *typelink[], *etypelink[];
	static int32 first = 1;
	ret.array = (byte*)typelink;
	ret.len = etypelink - typelink;
	ret.cap = ret.len;
	FLUSH(&ret);
}

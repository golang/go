// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <libc.h>
#include "go.h"
#include "../ld/textflag.h"
#include "../../runtime/mgc0.h"
#include "../../runtime/typekind.h"

/*
 * runtime interface and reflection data structures
 */

static	NodeList*	signatlist;
static	Sym*	dtypesym(Type*);
static	Sym*	weaktypesym(Type*);
static	Sym*	dalgsym(Type*);
static	int	usegcprog(Type*);
static	void	gengcprog(Type*, Sym**, Sym**);
static	void	gengcmask(Type*, uint8[16]);

static int
sigcmp(Sig *a, Sig *b)
{
	int i;

	i = strcmp(a->name, b->name);
	if(i != 0)
		return i;
	if(a->pkg == b->pkg)
		return 0;
	if(a->pkg == nil)
		return -1;
	if(b->pkg == nil)
		return +1;
	return strcmp(a->pkg->path->s, b->pkg->path->s);
}

static Sig*
lsort(Sig *l, int(*f)(Sig*, Sig*))
{
	Sig *l1, *l2, *le;

	if(l == 0 || l->link == 0)
		return l;

	l1 = l;
	l2 = l;
	for(;;) {
		l2 = l2->link;
		if(l2 == 0)
			break;
		l2 = l2->link;
		if(l2 == 0)
			break;
		l1 = l1->link;
	}

	l2 = l1->link;
	l1->link = 0;
	l1 = lsort(l, f);
	l2 = lsort(l2, f);

	/* set up lead element */
	if((*f)(l1, l2) < 0) {
		l = l1;
		l1 = l1->link;
	} else {
		l = l2;
		l2 = l2->link;
	}
	le = l;

	for(;;) {
		if(l1 == 0) {
			while(l2) {
				le->link = l2;
				le = l2;
				l2 = l2->link;
			}
			le->link = 0;
			break;
		}
		if(l2 == 0) {
			while(l1) {
				le->link = l1;
				le = l1;
				l1 = l1->link;
			}
			break;
		}
		if((*f)(l1, l2) < 0) {
			le->link = l1;
			le = l1;
			l1 = l1->link;
		} else {
			le->link = l2;
			le = l2;
			l2 = l2->link;
		}
	}
	le->link = 0;
	return l;
}

// Builds a type respresenting a Bucket structure for
// the given map type.  This type is not visible to users -
// we include only enough information to generate a correct GC
// program for it.
// Make sure this stays in sync with ../../runtime/hashmap.c!
enum {
	BUCKETSIZE = 8,
	MAXKEYSIZE = 128,
	MAXVALSIZE = 128,
};

static Type*
mapbucket(Type *t)
{
	Type *keytype, *valtype;
	Type *bucket;
	Type *overflowfield, *keysfield, *valuesfield;
	int32 offset;

	if(t->bucket != T)
		return t->bucket;

	keytype = t->down;
	valtype = t->type;
	dowidth(keytype);
	dowidth(valtype);
	if(keytype->width > MAXKEYSIZE)
		keytype = ptrto(keytype);
	if(valtype->width > MAXVALSIZE)
		valtype = ptrto(valtype);

	bucket = typ(TSTRUCT);
	bucket->noalg = 1;

	// The first field is: uint8 topbits[BUCKETSIZE].
	// We don't need to encode it as GC doesn't care about it.
	offset = BUCKETSIZE * 1;

	keysfield = typ(TFIELD);
	keysfield->type = typ(TARRAY);
	keysfield->type->type = keytype;
	keysfield->type->bound = BUCKETSIZE;
	keysfield->type->width = BUCKETSIZE * keytype->width;
	keysfield->width = offset;
	keysfield->sym = mal(sizeof(Sym));
	keysfield->sym->name = "keys";
	offset += BUCKETSIZE * keytype->width;

	valuesfield = typ(TFIELD);
	valuesfield->type = typ(TARRAY);
	valuesfield->type->type = valtype;
	valuesfield->type->bound = BUCKETSIZE;
	valuesfield->type->width = BUCKETSIZE * valtype->width;
	valuesfield->width = offset;
	valuesfield->sym = mal(sizeof(Sym));
	valuesfield->sym->name = "values";
	offset += BUCKETSIZE * valtype->width;

	overflowfield = typ(TFIELD);
	overflowfield->type = ptrto(bucket);
	overflowfield->width = offset;         // "width" is offset in structure
	overflowfield->sym = mal(sizeof(Sym)); // not important but needs to be set to give this type a name
	overflowfield->sym->name = "overflow";
	offset += widthptr;
	
	// Pad to the native integer alignment.
	// This is usually the same as widthptr; the exception (as usual) is nacl/amd64.
	if(widthreg > widthptr)
		offset += widthreg - widthptr;

	// link up fields
	bucket->type = keysfield;
	keysfield->down = valuesfield;
	valuesfield->down = overflowfield;
	overflowfield->down = T;

	bucket->width = offset;
	bucket->local = t->local;
	t->bucket = bucket;
	bucket->map = t;
	return bucket;
}

// Builds a type respresenting a Hmap structure for
// the given map type.  This type is not visible to users -
// we include only enough information to generate a correct GC
// program for it.
// Make sure this stays in sync with ../../runtime/hashmap.go!
static Type*
hmap(Type *t)
{
	Type *h, *bucket;
	Type *bucketsfield, *oldbucketsfield;
	int32 offset;

	if(t->hmap != T)
		return t->hmap;

	bucket = mapbucket(t);
	h = typ(TSTRUCT);
	h->noalg = 1;

	offset = widthint; // count
	offset += 4;       // flags
	offset += 4;       // hash0
	offset += 1;       // B
	offset = (offset + widthptr - 1) / widthptr * widthptr;
	
	bucketsfield = typ(TFIELD);
	bucketsfield->type = ptrto(bucket);
	bucketsfield->width = offset;
	bucketsfield->sym = mal(sizeof(Sym));
	bucketsfield->sym->name = "buckets";
	offset += widthptr;

	oldbucketsfield = typ(TFIELD);
	oldbucketsfield->type = ptrto(bucket);
	oldbucketsfield->width = offset;
	oldbucketsfield->sym = mal(sizeof(Sym));
	oldbucketsfield->sym->name = "oldbuckets";
	offset += widthptr;

	offset += widthptr; // nevacuate (last field in Hmap)

	// link up fields
	h->type = bucketsfield;
	bucketsfield->down = oldbucketsfield;
	oldbucketsfield->down = T;

	h->width = offset;
	h->local = t->local;
	t->hmap = h;
	h->map = t;
	return h;
}

Type*
hiter(Type *t)
{
	int32 n, off;
	Type *field[7];
	Type *i;

	if(t->hiter != T)
		return t->hiter;

	// build a struct:
	// hash_iter {
	//    key *Key
	//    val *Value
	//    t *MapType
	//    h *Hmap
	//    buckets *Bucket
	//    bptr *Bucket
	//    other [4]uintptr
	// }
	// must match ../../runtime/hashmap.c:hash_iter.
	field[0] = typ(TFIELD);
	field[0]->type = ptrto(t->down);
	field[0]->sym = mal(sizeof(Sym));
	field[0]->sym->name = "key";
	
	field[1] = typ(TFIELD);
	field[1]->type = ptrto(t->type);
	field[1]->sym = mal(sizeof(Sym));
	field[1]->sym->name = "val";
	
	field[2] = typ(TFIELD);
	field[2]->type = ptrto(types[TUINT8]); // TODO: is there a Type type?
	field[2]->sym = mal(sizeof(Sym));
	field[2]->sym->name = "t";
	
	field[3] = typ(TFIELD);
	field[3]->type = ptrto(hmap(t));
	field[3]->sym = mal(sizeof(Sym));
	field[3]->sym->name = "h";
	
	field[4] = typ(TFIELD);
	field[4]->type = ptrto(mapbucket(t));
	field[4]->sym = mal(sizeof(Sym));
	field[4]->sym->name = "buckets";
	
	field[5] = typ(TFIELD);
	field[5]->type = ptrto(mapbucket(t));
	field[5]->sym = mal(sizeof(Sym));
	field[5]->sym->name = "bptr";
	
	// all other non-pointer fields
	field[6] = typ(TFIELD);
	field[6]->type = typ(TARRAY);
	field[6]->type->type = types[TUINTPTR];
	field[6]->type->bound = 4;
	field[6]->type->width = 4 * widthptr;
	field[6]->sym = mal(sizeof(Sym));
	field[6]->sym->name = "other";
	
	// build iterator struct holding the above fields
	i = typ(TSTRUCT);
	i->noalg = 1;
	i->type = field[0];
	off = 0;
	for(n = 0; n < 6; n++) {
		field[n]->down = field[n+1];
		field[n]->width = off;
		off += field[n]->type->width;
	}
	field[6]->down = T;
	off += field[6]->type->width;
	if(off != 10 * widthptr)
		yyerror("hash_iter size not correct %d %d", off, 10 * widthptr);
	t->hiter = i;
	i->map = t;
	return i;
}

/*
 * f is method type, with receiver.
 * return function type, receiver as first argument (or not).
 */
Type*
methodfunc(Type *f, Type *receiver)
{
	NodeList *in, *out;
	Node *d;
	Type *t;

	in = nil;
	if(receiver) {
		d = nod(ODCLFIELD, N, N);
		d->type = receiver;
		in = list(in, d);
	}
	for(t=getinargx(f)->type; t; t=t->down) {
		d = nod(ODCLFIELD, N, N);
		d->type = t->type;
		d->isddd = t->isddd;
		in = list(in, d);
	}

	out = nil;
	for(t=getoutargx(f)->type; t; t=t->down) {
		d = nod(ODCLFIELD, N, N);
		d->type = t->type;
		out = list(out, d);
	}

	t = functype(N, in, out);
	if(f->nname) {
		// Link to name of original method function.
		t->nname = f->nname;
	}
	return t;
}

/*
 * return methods of non-interface type t, sorted by name.
 * generates stub functions as needed.
 */
static Sig*
methods(Type *t)
{
	Type *f, *mt, *it, *this;
	Sig *a, *b;
	Sym *method;

	// method type
	mt = methtype(t, 0);
	if(mt == T)
		return nil;
	expandmeth(mt);

	// type stored in interface word
	it = t;
	if(!isdirectiface(it))
		it = ptrto(t);

	// make list of methods for t,
	// generating code if necessary.
	a = nil;
	for(f=mt->xmethod; f; f=f->down) {
		if(f->etype != TFIELD)
			fatal("methods: not field %T", f);
		if (f->type->etype != TFUNC || f->type->thistuple == 0)
			fatal("non-method on %T method %S %T\n", mt, f->sym, f);
		if (!getthisx(f->type)->type)
			fatal("receiver with no type on %T method %S %T\n", mt, f->sym, f);
		if(f->nointerface)
			continue;

		method = f->sym;
		if(method == nil)
			continue;

		// get receiver type for this particular method.
		// if pointer receiver but non-pointer t and
		// this is not an embedded pointer inside a struct,
		// method does not apply.
		this = getthisx(f->type)->type->type;
		if(isptr[this->etype] && this->type == t)
			continue;
		if(isptr[this->etype] && !isptr[t->etype]
		&& f->embedded != 2 && !isifacemethod(f->type))
			continue;

		b = mal(sizeof(*b));
		b->link = a;
		a = b;

		a->name = method->name;
		if(!exportname(method->name)) {
			if(method->pkg == nil)
				fatal("methods: missing package");
			a->pkg = method->pkg;
		}
		a->isym = methodsym(method, it, 1);
		a->tsym = methodsym(method, t, 0);
		a->type = methodfunc(f->type, t);
		a->mtype = methodfunc(f->type, nil);

		if(!(a->isym->flags & SymSiggen)) {
			a->isym->flags |= SymSiggen;
			if(!eqtype(this, it) || this->width < types[tptr]->width) {
				compiling_wrappers = 1;
				genwrapper(it, f, a->isym, 1);
				compiling_wrappers = 0;
			}
		}

		if(!(a->tsym->flags & SymSiggen)) {
			a->tsym->flags |= SymSiggen;
			if(!eqtype(this, t)) {
				compiling_wrappers = 1;
				genwrapper(t, f, a->tsym, 0);
				compiling_wrappers = 0;
			}
		}
	}

	return lsort(a, sigcmp);
}

/*
 * return methods of interface type t, sorted by name.
 */
static Sig*
imethods(Type *t)
{
	Sig *a, *all, *last;
	Type *f;
	Sym *method, *isym;

	all = nil;
	last = nil;
	for(f=t->type; f; f=f->down) {
		if(f->etype != TFIELD)
			fatal("imethods: not field");
		if(f->type->etype != TFUNC || f->sym == nil)
			continue;
		method = f->sym;
		a = mal(sizeof(*a));
		a->name = method->name;
		if(!exportname(method->name)) {
			if(method->pkg == nil)
				fatal("imethods: missing package");
			a->pkg = method->pkg;
		}
		a->mtype = f->type;
		a->offset = 0;
		a->type = methodfunc(f->type, nil);

		if(last && sigcmp(last, a) >= 0)
			fatal("sigcmp vs sortinter %s %s", last->name, a->name);
		if(last == nil)
			all = a;
		else
			last->link = a;
		last = a;

		// Compiler can only refer to wrappers for non-blank methods.
		if(isblanksym(method))
			continue;

		// NOTE(rsc): Perhaps an oversight that
		// IfaceType.Method is not in the reflect data.
		// Generate the method body, so that compiled
		// code can refer to it.
		isym = methodsym(method, t, 0);
		if(!(isym->flags & SymSiggen)) {
			isym->flags |= SymSiggen;
			genwrapper(t, f, isym, 0);
		}
	}
	return all;
}

static void
dimportpath(Pkg *p)
{
	static Pkg *gopkg;
	char *nam;
	Node *n;

	if(p->pathsym != S)
		return;

	if(gopkg == nil) {
		gopkg = mkpkg(strlit("go"));
		gopkg->name = "go";
	}
	nam = smprint("importpath.%s.", p->prefix);

	n = nod(ONAME, N, N);
	n->sym = pkglookup(nam, gopkg);
	free(nam);
	n->class = PEXTERN;
	n->xoffset = 0;
	p->pathsym = n->sym;

	gdatastring(n, p->path);
	ggloblsym(n->sym, types[TSTRING]->width, DUPOK|RODATA);
}

static int
dgopkgpath(Sym *s, int ot, Pkg *pkg)
{
	if(pkg == nil)
		return dgostringptr(s, ot, nil);

	// Emit reference to go.importpath.""., which 6l will
	// rewrite using the correct import path.  Every package
	// that imports this one directly defines the symbol.
	if(pkg == localpkg) {
		static Sym *ns;

		if(ns == nil)
			ns = pkglookup("importpath.\"\".", mkpkg(strlit("go")));
		return dsymptr(s, ot, ns, 0);
	}

	dimportpath(pkg);
	return dsymptr(s, ot, pkg->pathsym, 0);
}

/*
 * uncommonType
 * ../../runtime/type.go:/uncommonType
 */
static int
dextratype(Sym *sym, int off, Type *t, int ptroff)
{
	int ot, n;
	Sym *s;
	Sig *a, *m;

	m = methods(t);
	if(t->sym == nil && m == nil)
		return off;

	// fill in *extraType pointer in header
	off = rnd(off, widthptr);
	dsymptr(sym, ptroff, sym, off);

	n = 0;
	for(a=m; a; a=a->link) {
		dtypesym(a->type);
		n++;
	}

	ot = off;
	s = sym;
	if(t->sym) {
		ot = dgostringptr(s, ot, t->sym->name);
		if(t != types[t->etype] && t != errortype)
			ot = dgopkgpath(s, ot, t->sym->pkg);
		else
			ot = dgostringptr(s, ot, nil);
	} else {
		ot = dgostringptr(s, ot, nil);
		ot = dgostringptr(s, ot, nil);
	}

	// slice header
	ot = dsymptr(s, ot, s, ot + widthptr + 2*widthint);
	ot = duintxx(s, ot, n, widthint);
	ot = duintxx(s, ot, n, widthint);

	// methods
	for(a=m; a; a=a->link) {
		// method
		// ../../runtime/type.go:/method
		ot = dgostringptr(s, ot, a->name);
		ot = dgopkgpath(s, ot, a->pkg);
		ot = dsymptr(s, ot, dtypesym(a->mtype), 0);
		ot = dsymptr(s, ot, dtypesym(a->type), 0);
		if(a->isym)
			ot = dsymptr(s, ot, a->isym, 0);
		else
			ot = duintptr(s, ot, 0);
		if(a->tsym)
			ot = dsymptr(s, ot, a->tsym, 0);
		else
			ot = duintptr(s, ot, 0);
	}

	return ot;
}

static int
kinds[] =
{
	[TINT]		= KindInt,
	[TUINT]		= KindUint,
	[TINT8]		= KindInt8,
	[TUINT8]	= KindUint8,
	[TINT16]	= KindInt16,
	[TUINT16]	= KindUint16,
	[TINT32]	= KindInt32,
	[TUINT32]	= KindUint32,
	[TINT64]	= KindInt64,
	[TUINT64]	= KindUint64,
	[TUINTPTR]	= KindUintptr,
	[TFLOAT32]	= KindFloat32,
	[TFLOAT64]	= KindFloat64,
	[TBOOL]		= KindBool,
	[TSTRING]		= KindString,
	[TPTR32]		= KindPtr,
	[TPTR64]		= KindPtr,
	[TSTRUCT]	= KindStruct,
	[TINTER]		= KindInterface,
	[TCHAN]		= KindChan,
	[TMAP]		= KindMap,
	[TARRAY]		= KindArray,
	[TFUNC]		= KindFunc,
	[TCOMPLEX64]	= KindComplex64,
	[TCOMPLEX128]	= KindComplex128,
	[TUNSAFEPTR]	= KindUnsafePointer,
};

int
haspointers(Type *t)
{
	Type *t1;
	int ret;

	if(t->haspointers != 0)
		return t->haspointers - 1;

	switch(t->etype) {
	case TINT:
	case TUINT:
	case TINT8:
	case TUINT8:
	case TINT16:
	case TUINT16:
	case TINT32:
	case TUINT32:
	case TINT64:
	case TUINT64:
	case TUINTPTR:
	case TFLOAT32:
	case TFLOAT64:
	case TCOMPLEX64:
	case TCOMPLEX128:
	case TBOOL:
		ret = 0;
		break;
	case TARRAY:
		if(t->bound < 0) {	// slice
			ret = 1;
			break;
		}
		if(t->bound == 0) {	// empty array
			ret = 0;
			break;
		}
		ret = haspointers(t->type);
		break;
	case TSTRUCT:
		ret = 0;
		for(t1=t->type; t1!=T; t1=t1->down) {
			if(haspointers(t1->type)) {
				ret = 1;
				break;
			}
		}
		break;
	case TSTRING:
	case TPTR32:
	case TPTR64:
	case TUNSAFEPTR:
	case TINTER:
	case TCHAN:
	case TMAP:
	case TFUNC:
	default:
		ret = 1;
		break;
	}
	
	t->haspointers = 1+ret;
	return ret;
}

/*
 * commonType
 * ../../runtime/type.go:/commonType
 */
static int
dcommontype(Sym *s, int ot, Type *t)
{
	int i, alg, sizeofAlg, gcprog;
	Sym *sptr, *algsym, *zero, *gcprog0, *gcprog1, *sbits;
	uint8 gcmask[16];
	static Sym *algarray;
	uint64 x1, x2;
	char *p;
	
	if(ot != 0)
		fatal("dcommontype %d", ot);

	sizeofAlg = 2*widthptr;
	if(algarray == nil)
		algarray = pkglookup("algarray", runtimepkg);
	dowidth(t);
	alg = algtype(t);
	algsym = S;
	if(alg < 0)
		algsym = dalgsym(t);

	if(t->sym != nil && !isptr[t->etype])
		sptr = dtypesym(ptrto(t));
	else
		sptr = weaktypesym(ptrto(t));

	// All (non-reflect-allocated) Types share the same zero object.
	// Each place in the compiler where a pointer to the zero object
	// might be returned by a runtime call (map access return value,
	// 2-arg type cast) declares the size of the zerovalue it needs.
	// The linker magically takes the max of all the sizes.
	zero = pkglookup("zerovalue", runtimepkg);

	// We use size 0 here so we get the pointer to the zero value,
	// but don't allocate space for the zero value unless we need it.
	// TODO: how do we get this symbol into bss?  We really want
	// a read-only bss, but I don't think such a thing exists.

	// ../../pkg/reflect/type.go:/^type.commonType
	// actual type structure
	//	type commonType struct {
	//		size          uintptr
	//		hash          uint32
	//		_             uint8
	//		align         uint8
	//		fieldAlign    uint8
	//		kind          uint8
	//		alg           unsafe.Pointer
	//		gc            unsafe.Pointer
	//		string        *string
	//		*extraType
	//		ptrToThis     *Type
	//		zero          unsafe.Pointer
	//	}
	ot = duintptr(s, ot, t->width);
	ot = duint32(s, ot, typehash(t));
	ot = duint8(s, ot, 0);	// unused

	// runtime (and common sense) expects alignment to be a power of two.
	i = t->align;
	if(i == 0)
		i = 1;
	if((i&(i-1)) != 0)
		fatal("invalid alignment %d for %T", t->align, t);
	ot = duint8(s, ot, t->align);	// align
	ot = duint8(s, ot, t->align);	// fieldAlign

	gcprog = usegcprog(t);
	i = kinds[t->etype];
	if(t->etype == TARRAY && t->bound < 0)
		i = KindSlice;
	if(!haspointers(t))
		i |= KindNoPointers;
	if(isdirectiface(t))
		i |= KindDirectIface;
	if(gcprog)
		i |= KindGCProg;
	ot = duint8(s, ot, i);  // kind
	if(alg >= 0)
		ot = dsymptr(s, ot, algarray, alg*sizeofAlg);
	else
		ot = dsymptr(s, ot, algsym, 0);
	// gc
	if(gcprog) {
		gengcprog(t, &gcprog0, &gcprog1);
		if(gcprog0 != S)
			ot = dsymptr(s, ot, gcprog0, 0);
		else
			ot = duintptr(s, ot, 0);
		ot = dsymptr(s, ot, gcprog1, 0);
	} else {
		gengcmask(t, gcmask);
		x1 = 0;
		for(i=0; i<8; i++)
			x1 = x1<<8 | gcmask[i];
		if(widthptr == 4) {
			p = smprint("gcbits.%#016llux", x1);
		} else {
			x2 = 0;
			for(i=0; i<8; i++)
				x2 = x2<<8 | gcmask[i+8];
			p = smprint("gcbits.%#016llux%016llux", x1, x2);
		}
		sbits = pkglookup(p, runtimepkg);
		if((sbits->flags & SymUniq) == 0) {
			sbits->flags |= SymUniq;
			for(i = 0; i < 2*widthptr; i++)
				duint8(sbits, i, gcmask[i]);
			ggloblsym(sbits, 2*widthptr, DUPOK|RODATA);
		}
		ot = dsymptr(s, ot, sbits, 0);
		ot = duintptr(s, ot, 0);
	}
	p = smprint("%-uT", t);
	//print("dcommontype: %s\n", p);
	ot = dgostringptr(s, ot, p);	// string
	free(p);

	// skip pointer to extraType,
	// which follows the rest of this type structure.
	// caller will fill in if needed.
	// otherwise linker will assume 0.
	ot += widthptr;

	ot = dsymptr(s, ot, sptr, 0);  // ptrto type
	ot = dsymptr(s, ot, zero, 0);  // ptr to zero value
	return ot;
}

Sym*
typesym(Type *t)
{
	char *p;
	Sym *s;

	p = smprint("%-T", t);
	s = pkglookup(p, typepkg);
	//print("typesym: %s -> %+S\n", p, s);
	free(p);
	return s;
}

Sym*
tracksym(Type *t)
{
	char *p;
	Sym *s;

	p = smprint("%-T.%s", t->outer, t->sym->name);
	s = pkglookup(p, trackpkg);
	free(p);
	return s;
}

Sym*
typelinksym(Type *t)
{
	char *p;
	Sym *s;

	// %-uT is what the generated Type's string field says.
	// It uses (ambiguous) package names instead of import paths.
	// %-T is the complete, unambiguous type name.
	// We want the types to end up sorted by string field,
	// so use that first in the name, and then add :%-T to
	// disambiguate. The names are a little long but they are
	// discarded by the linker and do not end up in the symbol
	// table of the final binary.
	p = smprint("%-uT/%-T", t, t);
	s = pkglookup(p, typelinkpkg);
	//print("typelinksym: %s -> %+S\n", p, s);
	free(p);
	return s;
}

Sym*
typesymprefix(char *prefix, Type *t)
{
	char *p;
	Sym *s;

	p = smprint("%s.%-T", prefix, t);
	s = pkglookup(p, typepkg);
	//print("algsym: %s -> %+S\n", p, s);
	free(p);
	return s;
}

Sym*
typenamesym(Type *t)
{
	Sym *s;
	Node *n;

	if(t == T || (isptr[t->etype] && t->type == T) || isideal(t))
		fatal("typename %T", t);
	s = typesym(t);
	if(s->def == N) {
		n = nod(ONAME, N, N);
		n->sym = s;
		n->type = types[TUINT8];
		n->addable = 1;
		n->ullman = 1;
		n->class = PEXTERN;
		n->xoffset = 0;
		n->typecheck = 1;
		s->def = n;

		signatlist = list(signatlist, typenod(t));
	}
	return s->def->sym;
}

Node*
typename(Type *t)
{
	Sym *s;
	Node *n;

	s = typenamesym(t);
	n = nod(OADDR, s->def, N);
	n->type = ptrto(s->def->type);
	n->addable = 1;
	n->ullman = 2;
	n->typecheck = 1;
	return n;
}

static Sym*
weaktypesym(Type *t)
{
	char *p;
	Sym *s;

	p = smprint("%-T", t);
	s = pkglookup(p, weaktypepkg);
	//print("weaktypesym: %s -> %+S\n", p, s);
	free(p);
	return s;
}

/*
 * Returns 1 if t has a reflexive equality operator.
 * That is, if x==x for all x of type t.
 */
static int
isreflexive(Type *t)
{
	Type *t1;
	switch(t->etype) {
		case TBOOL:
		case TINT:
		case TUINT:
		case TINT8:
		case TUINT8:
		case TINT16:
		case TUINT16:
		case TINT32:
		case TUINT32:
		case TINT64:
		case TUINT64:
		case TUINTPTR:
		case TPTR32:
		case TPTR64:
		case TUNSAFEPTR:
		case TSTRING:
		case TCHAN:
			return 1;
		case TFLOAT32:
		case TFLOAT64:
		case TCOMPLEX64:
		case TCOMPLEX128:
		case TINTER:
			return 0;
		case TARRAY:
			if(isslice(t))
				fatal("slice can't be a map key: %T", t);
			return isreflexive(t->type);
		case TSTRUCT:
			for(t1=t->type; t1!=T; t1=t1->down) {
				if(!isreflexive(t1->type))
					return 0;
			}
			return 1;
		default:
			fatal("bad type for map key: %T", t);
			return 0;
	}
}

static Sym*
dtypesym(Type *t)
{
	int ot, xt, n, isddd, dupok;
	Sym *s, *s1, *s2, *s3, *s4, *slink;
	Sig *a, *m;
	Type *t1, *tbase, *t2;

	// Replace byte, rune aliases with real type.
	// They've been separate internally to make error messages
	// better, but we have to merge them in the reflect tables.
	if(t == bytetype || t == runetype)
		t = types[t->etype];

	if(isideal(t))
		fatal("dtypesym %T", t);

	s = typesym(t);
	if(s->flags & SymSiggen)
		return s;
	s->flags |= SymSiggen;

	// special case (look for runtime below):
	// when compiling package runtime,
	// emit the type structures for int, float, etc.
	tbase = t;
	if(isptr[t->etype] && t->sym == S && t->type->sym != S)
		tbase = t->type;
	dupok = 0;
	if(tbase->sym == S)
		dupok = DUPOK;

	if(compiling_runtime &&
			(tbase == types[tbase->etype] ||
			tbase == bytetype ||
			tbase == runetype ||
			tbase == errortype)) { // int, float, etc
		goto ok;
	}

	// named types from other files are defined only by those files
	if(tbase->sym && !tbase->local)
		return s;
	if(isforw[tbase->etype])
		return s;

ok:
	ot = 0;
	xt = 0;
	switch(t->etype) {
	default:
		ot = dcommontype(s, ot, t);
		xt = ot - 3*widthptr;
		break;

	case TARRAY:
		if(t->bound >= 0) {
			// ../../runtime/type.go:/ArrayType
			s1 = dtypesym(t->type);
			t2 = typ(TARRAY);
			t2->type = t->type;
			t2->bound = -1;  // slice
			s2 = dtypesym(t2);
			ot = dcommontype(s, ot, t);
			xt = ot - 3*widthptr;
			ot = dsymptr(s, ot, s1, 0);
			ot = dsymptr(s, ot, s2, 0);
			ot = duintptr(s, ot, t->bound);
		} else {
			// ../../runtime/type.go:/SliceType
			s1 = dtypesym(t->type);
			ot = dcommontype(s, ot, t);
			xt = ot - 3*widthptr;
			ot = dsymptr(s, ot, s1, 0);
		}
		break;

	case TCHAN:
		// ../../runtime/type.go:/ChanType
		s1 = dtypesym(t->type);
		ot = dcommontype(s, ot, t);
		xt = ot - 3*widthptr;
		ot = dsymptr(s, ot, s1, 0);
		ot = duintptr(s, ot, t->chan);
		break;

	case TFUNC:
		for(t1=getthisx(t)->type; t1; t1=t1->down)
			dtypesym(t1->type);
		isddd = 0;
		for(t1=getinargx(t)->type; t1; t1=t1->down) {
			isddd = t1->isddd;
			dtypesym(t1->type);
		}
		for(t1=getoutargx(t)->type; t1; t1=t1->down)
			dtypesym(t1->type);

		ot = dcommontype(s, ot, t);
		xt = ot - 3*widthptr;
		ot = duint8(s, ot, isddd);

		// two slice headers: in and out.
		ot = rnd(ot, widthptr);
		ot = dsymptr(s, ot, s, ot+2*(widthptr+2*widthint));
		n = t->thistuple + t->intuple;
		ot = duintxx(s, ot, n, widthint);
		ot = duintxx(s, ot, n, widthint);
		ot = dsymptr(s, ot, s, ot+1*(widthptr+2*widthint)+n*widthptr);
		ot = duintxx(s, ot, t->outtuple, widthint);
		ot = duintxx(s, ot, t->outtuple, widthint);

		// slice data
		for(t1=getthisx(t)->type; t1; t1=t1->down, n++)
			ot = dsymptr(s, ot, dtypesym(t1->type), 0);
		for(t1=getinargx(t)->type; t1; t1=t1->down, n++)
			ot = dsymptr(s, ot, dtypesym(t1->type), 0);
		for(t1=getoutargx(t)->type; t1; t1=t1->down, n++)
			ot = dsymptr(s, ot, dtypesym(t1->type), 0);
		break;

	case TINTER:
		m = imethods(t);
		n = 0;
		for(a=m; a; a=a->link) {
			dtypesym(a->type);
			n++;
		}

		// ../../runtime/type.go:/InterfaceType
		ot = dcommontype(s, ot, t);
		xt = ot - 3*widthptr;
		ot = dsymptr(s, ot, s, ot+widthptr+2*widthint);
		ot = duintxx(s, ot, n, widthint);
		ot = duintxx(s, ot, n, widthint);
		for(a=m; a; a=a->link) {
			// ../../runtime/type.go:/imethod
			ot = dgostringptr(s, ot, a->name);
			ot = dgopkgpath(s, ot, a->pkg);
			ot = dsymptr(s, ot, dtypesym(a->type), 0);
		}
		break;

	case TMAP:
		// ../../runtime/type.go:/MapType
		s1 = dtypesym(t->down);
		s2 = dtypesym(t->type);
		s3 = dtypesym(mapbucket(t));
		s4 = dtypesym(hmap(t));
		ot = dcommontype(s, ot, t);
		xt = ot - 3*widthptr;
		ot = dsymptr(s, ot, s1, 0);
		ot = dsymptr(s, ot, s2, 0);
		ot = dsymptr(s, ot, s3, 0);
		ot = dsymptr(s, ot, s4, 0);
		if(t->down->width > MAXKEYSIZE) {
			ot = duint8(s, ot, widthptr);
			ot = duint8(s, ot, 1); // indirect
		} else {
			ot = duint8(s, ot, t->down->width);
			ot = duint8(s, ot, 0); // not indirect
		}
		if(t->type->width > MAXVALSIZE) {
			ot = duint8(s, ot, widthptr);
			ot = duint8(s, ot, 1); // indirect
		} else {
			ot = duint8(s, ot, t->type->width);
			ot = duint8(s, ot, 0); // not indirect
		}
		ot = duint16(s, ot, mapbucket(t)->width);
                ot = duint8(s, ot, isreflexive(t->down));
		break;

	case TPTR32:
	case TPTR64:
		if(t->type->etype == TANY) {
			// ../../runtime/type.go:/UnsafePointerType
			ot = dcommontype(s, ot, t);
			break;
		}
		// ../../runtime/type.go:/PtrType
		s1 = dtypesym(t->type);
		ot = dcommontype(s, ot, t);
		xt = ot - 3*widthptr;
		ot = dsymptr(s, ot, s1, 0);
		break;

	case TSTRUCT:
		// ../../runtime/type.go:/StructType
		// for security, only the exported fields.
		n = 0;
		for(t1=t->type; t1!=T; t1=t1->down) {
			dtypesym(t1->type);
			n++;
		}
		ot = dcommontype(s, ot, t);
		xt = ot - 3*widthptr;
		ot = dsymptr(s, ot, s, ot+widthptr+2*widthint);
		ot = duintxx(s, ot, n, widthint);
		ot = duintxx(s, ot, n, widthint);
		for(t1=t->type; t1!=T; t1=t1->down) {
			// ../../runtime/type.go:/structField
			if(t1->sym && !t1->embedded) {
				ot = dgostringptr(s, ot, t1->sym->name);
				if(exportname(t1->sym->name))
					ot = dgostringptr(s, ot, nil);
				else
					ot = dgopkgpath(s, ot, t1->sym->pkg);
			} else {
				ot = dgostringptr(s, ot, nil);
				if(t1->type->sym != S && t1->type->sym->pkg == builtinpkg)
					ot = dgopkgpath(s, ot, localpkg);
				else
					ot = dgostringptr(s, ot, nil);
			}
			ot = dsymptr(s, ot, dtypesym(t1->type), 0);
			ot = dgostrlitptr(s, ot, t1->note);
			ot = duintptr(s, ot, t1->width);	// field offset
		}
		break;
	}
	ot = dextratype(s, ot, t, xt);
	ggloblsym(s, ot, dupok|RODATA);

	// generate typelink.foo pointing at s = type.foo.
	// The linker will leave a table of all the typelinks for
	// types in the binary, so reflect can find them.
	// We only need the link for unnamed composites that
	// we want be able to find.
	if(t->sym == S) {
		switch(t->etype) {
		case TARRAY:
		case TCHAN:
		case TMAP:
			slink = typelinksym(t);
			dsymptr(slink, 0, s, 0);
			ggloblsym(slink, widthptr, dupok|RODATA);
		}
	}

	return s;
}

void
dumptypestructs(void)
{
	int i;
	NodeList *l;
	Node *n;
	Type *t;
	Pkg *p;

	// copy types from externdcl list to signatlist
	for(l=externdcl; l; l=l->next) {
		n = l->n;
		if(n->op != OTYPE)
			continue;
		signatlist = list(signatlist, n);
	}

	// process signatlist
	for(l=signatlist; l; l=l->next) {
		n = l->n;
		if(n->op != OTYPE)
			continue;
		t = n->type;
		dtypesym(t);
		if(t->sym)
			dtypesym(ptrto(t));
	}

	// generate import strings for imported packages
	for(i=0; i<nelem(phash); i++)
		for(p=phash[i]; p; p=p->link)
			if(p->direct)
				dimportpath(p);

	// do basic types if compiling package runtime.
	// they have to be in at least one package,
	// and runtime is always loaded implicitly,
	// so this is as good as any.
	// another possible choice would be package main,
	// but using runtime means fewer copies in .6 files.
	if(compiling_runtime) {
		for(i=1; i<=TBOOL; i++)
			dtypesym(ptrto(types[i]));
		dtypesym(ptrto(types[TSTRING]));
		dtypesym(ptrto(types[TUNSAFEPTR]));

		// emit type structs for error and func(error) string.
		// The latter is the type of an auto-generated wrapper.
		dtypesym(ptrto(errortype));
		dtypesym(functype(nil,
			list1(nod(ODCLFIELD, N, typenod(errortype))),
			list1(nod(ODCLFIELD, N, typenod(types[TSTRING])))));

		// add paths for runtime and main, which 6l imports implicitly.
		dimportpath(runtimepkg);
		if(flag_race)
			dimportpath(racepkg);
		dimportpath(mkpkg(strlit("main")));
	}
}

static Sym*
dalgsym(Type *t)
{
	int ot;
	Sym *s, *hash, *hashfunc, *eq, *eqfunc;

	// dalgsym is only called for a type that needs an algorithm table,
	// which implies that the type is comparable (or else it would use ANOEQ).

	s = typesymprefix(".alg", t);
	hash = typesymprefix(".hash", t);
	genhash(hash, t);
	eq = typesymprefix(".eq", t);
	geneq(eq, t);

	// make Go funcs (closures) for calling hash and equal from Go
	hashfunc = typesymprefix(".hashfunc", t);
	dsymptr(hashfunc, 0, hash, 0);
	ggloblsym(hashfunc, widthptr, DUPOK|RODATA);
	eqfunc = typesymprefix(".eqfunc", t);
	dsymptr(eqfunc, 0, eq, 0);
	ggloblsym(eqfunc, widthptr, DUPOK|RODATA);

	// ../../runtime/alg.go:/typeAlg
	ot = 0;
	ot = dsymptr(s, ot, hashfunc, 0);
	ot = dsymptr(s, ot, eqfunc, 0);

	ggloblsym(s, ot, DUPOK|RODATA);
	return s;
}

static int
usegcprog(Type *t)
{
	vlong size, nptr;

	if(!haspointers(t))
		return 0;
	if(t->width == BADWIDTH)
		dowidth(t);
	// Calculate size of the unrolled GC mask.
	nptr = (t->width+widthptr-1)/widthptr;
	size = nptr;
	if(size%2)
		size *= 2;	// repeated
	size = size*gcBits/8;	// 4 bits per word
	// Decide whether to use unrolled GC mask or GC program.
	// We could use a more elaborate condition, but this seems to work well in practice.
	// For small objects GC program can't give significant reduction.
	// While large objects usually contain arrays; and even if it don't
	// the program uses 2-bits per word while mask uses 4-bits per word,
	// so the program is still smaller.
	return size > 2*widthptr;
}

// Generates sparse GC bitmask (4 bits per word).
static void
gengcmask(Type *t, uint8 gcmask[16])
{
	Bvec *vec;
	vlong xoffset, nptr, i, j;
	int  half;
	uint8 bits, *pos;

	memset(gcmask, 0, 16);
	if(!haspointers(t))
		return;

	// Generate compact mask as stacks use.
	xoffset = 0;
	vec = bvalloc(2*widthptr*8);
	twobitwalktype1(t, &xoffset, vec);

	// Unfold the mask for the GC bitmap format:
	// 4 bits per word, 2 high bits encode pointer info.
	pos = (uint8*)gcmask;
	nptr = (t->width+widthptr-1)/widthptr;
	half = 0;
	// If number of words is odd, repeat the mask.
	// This makes simpler handling of arrays in runtime.
	for(j=0; j<=(nptr%2); j++) {
		for(i=0; i<nptr; i++) {
			bits = bvget(vec, i*BitsPerPointer) | bvget(vec, i*BitsPerPointer+1)<<1;
			// Some fake types (e.g. Hmap) has missing fileds.
			// twobitwalktype1 generates BitsDead for that holes,
			// replace BitsDead with BitsScalar.
			if(bits == BitsDead)
				bits = BitsScalar;
			bits <<= 2;
			if(half)
				bits <<= 4;
			*pos |= bits;
			half = !half;
			if(!half)
				pos++;
		}
	}
}

// Helper object for generation of GC programs.
typedef struct ProgGen ProgGen;
struct ProgGen
{
	Sym*	s;
	int32	datasize;
	uint8	data[256/PointersPerByte];
	vlong	ot;
};

static void
proggeninit(ProgGen *g, Sym *s)
{
	g->s = s;
	g->datasize = 0;
	g->ot = 0;
	memset(g->data, 0, sizeof(g->data));
}

static void
proggenemit(ProgGen *g, uint8 v)
{
	g->ot = duint8(g->s, g->ot, v);
}

// Emits insData block from g->data.
static void
proggendataflush(ProgGen *g)
{
	int32 i, s;

	if(g->datasize == 0)
		return;
	proggenemit(g, insData);
	proggenemit(g, g->datasize);
	s = (g->datasize + PointersPerByte - 1)/PointersPerByte;
	for(i = 0; i < s; i++)
		proggenemit(g, g->data[i]);
	g->datasize = 0;
	memset(g->data, 0, sizeof(g->data));
}

static void
proggendata(ProgGen *g, uint8 d)
{
	g->data[g->datasize/PointersPerByte] |= d << ((g->datasize%PointersPerByte)*BitsPerPointer);
	g->datasize++;
	if(g->datasize == 255)
		proggendataflush(g);
}

// Skip v bytes due to alignment, etc.
static void
proggenskip(ProgGen *g, vlong off, vlong v)
{
	vlong i;

	for(i = off; i < off+v; i++) {
		if((i%widthptr) == 0)
			proggendata(g, BitsScalar);
	}
}

// Emit insArray instruction.
static void
proggenarray(ProgGen *g, vlong len)
{
	int32 i;

	proggendataflush(g);
	proggenemit(g, insArray);
	for(i = 0; i < widthptr; i++, len >>= 8)
		proggenemit(g, len);
}

static void
proggenarrayend(ProgGen *g)
{
	proggendataflush(g);
	proggenemit(g, insArrayEnd);
}

static vlong
proggenfini(ProgGen *g)
{
	proggendataflush(g);
	proggenemit(g, insEnd);
	return g->ot;
}

static void gengcprog1(ProgGen *g, Type *t, vlong *xoffset);

// Generates GC program for large types.
static void
gengcprog(Type *t, Sym **pgc0, Sym **pgc1)
{
	Sym *gc0, *gc1;
	vlong nptr, size, ot, xoffset;
	ProgGen g;

	nptr = (t->width+widthptr-1)/widthptr;
	size = nptr;
	if(size%2)
		size *= 2;	// repeated twice
	size = size*PointersPerByte/8;	// 4 bits per word
	size++;	// unroll flag in the beginning, used by runtime (see runtime.markallocated)
	// emity space in BSS for unrolled program
	*pgc0 = S;
	// Don't generate it if it's too large, runtime will unroll directly into GC bitmap.
	if(size <= MaxGCMask) {
		gc0 = typesymprefix(".gc", t);
		ggloblsym(gc0, size, DUPOK|NOPTR);
		*pgc0 = gc0;
	}

	// program in RODATA
	gc1 = typesymprefix(".gcprog", t);
	proggeninit(&g, gc1);
	xoffset = 0;
	gengcprog1(&g, t, &xoffset);
	ot = proggenfini(&g);
	ggloblsym(gc1, ot, DUPOK|RODATA);
	*pgc1 = gc1;
}

// Recursively walks type t and writes GC program into g.
static void
gengcprog1(ProgGen *g, Type *t, vlong *xoffset)
{
	vlong fieldoffset, i, o, n;
	Type *t1;

	switch(t->etype) {
	case TINT8:
	case TUINT8:
	case TINT16:
	case TUINT16:
	case TINT32:
	case TUINT32:
	case TINT64:
	case TUINT64:
	case TINT:
	case TUINT:
	case TUINTPTR:
	case TBOOL:
	case TFLOAT32:
	case TFLOAT64:
	case TCOMPLEX64:
	case TCOMPLEX128:
		proggenskip(g, *xoffset, t->width);
		*xoffset += t->width;
		break;
	case TPTR32:
	case TPTR64:
	case TUNSAFEPTR:
	case TFUNC:
	case TCHAN:
	case TMAP:
		proggendata(g, BitsPointer);
		*xoffset += t->width;
		break;
	case TSTRING:
		proggendata(g, BitsPointer);
		proggendata(g, BitsScalar);
		*xoffset += t->width;
		break;
	case TINTER:
		// Assuming IfacePointerOnly=1.
		proggendata(g, BitsPointer);
		proggendata(g, BitsPointer);
		*xoffset += t->width;
		break;
	case TARRAY:
		if(isslice(t)) {
			proggendata(g, BitsPointer);
			proggendata(g, BitsScalar);
			proggendata(g, BitsScalar);
		} else {
			t1 = t->type;
			if(t1->width == 0) {
				// ignore
			} if(t->bound <= 1 || t->bound*t1->width < 32*widthptr) {
				for(i = 0; i < t->bound; i++)
					gengcprog1(g, t1, xoffset);
			} else if(!haspointers(t1)) {
				n = t->width;
				n -= -*xoffset&(widthptr-1); // skip to next ptr boundary
				proggenarray(g, (n+widthptr-1)/widthptr);
				proggendata(g, BitsScalar);
				proggenarrayend(g);
				*xoffset -= (n+widthptr-1)/widthptr*widthptr - t->width;
			} else {
				proggenarray(g, t->bound);
				gengcprog1(g, t1, xoffset);
				*xoffset += (t->bound-1)*t1->width;
				proggenarrayend(g);
			}
		}
		break;
	case TSTRUCT:
		o = 0;
		for(t1 = t->type; t1 != T; t1 = t1->down) {
			fieldoffset = t1->width;
			proggenskip(g, *xoffset, fieldoffset - o);
			*xoffset += fieldoffset - o;
			gengcprog1(g, t1->type, xoffset);
			o = fieldoffset + t1->type->width;
		}
		proggenskip(g, *xoffset, t->width - o);
		*xoffset += t->width - o;
		break;
	default:
		fatal("gengcprog1: unexpected type, %T", t);
	}
}

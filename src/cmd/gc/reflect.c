// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <libc.h>
#include "go.h"

/*
 * runtime interface and reflection data structures
 */

static	NodeList*	signatlist;
static	Sym*	dtypesym(Type*);
static	Sym*	weaktypesym(Type*);
static	Sym*	dalgsym(Type*);

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
	if(it->width > widthptr)
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
				// Is okay to call genwrapper here always,
				// but we can generate more efficient code
				// using genembedtramp if all that is necessary
				// is a pointer adjustment and a JMP.
				if(isptr[it->etype] && isptr[this->etype]
				&& f->embedded && !isifacemethod(f->type))
					genembedtramp(it, f, a->isym, 1);
				else
					genwrapper(it, f, a->isym, 1);
			}
		}

		if(!(a->tsym->flags & SymSiggen)) {
			a->tsym->flags |= SymSiggen;
			if(!eqtype(this, t)) {
				if(isptr[t->etype] && isptr[this->etype]
				&& f->embedded && !isifacemethod(f->type))
					genembedtramp(t, f, a->tsym, 0);
				else
					genwrapper(t, f, a->tsym, 0);
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

		// Compiler can only refer to wrappers for
		// named interface types.
		if(t->sym == S)
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
	ggloblsym(n->sym, types[TSTRING]->width, 1, 1);
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
 * ../../pkg/runtime/type.go:/uncommonType
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
	ot = dsymptr(s, ot, s, ot + widthptr + 2*4);
	ot = duint32(s, ot, n);
	ot = duint32(s, ot, n);

	// methods
	for(a=m; a; a=a->link) {
		// method
		// ../../pkg/runtime/type.go:/method
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

enum {
	KindBool = 1,
	KindInt,
	KindInt8,
	KindInt16,
	KindInt32,
	KindInt64,
	KindUint,
	KindUint8,
	KindUint16,
	KindUint32,
	KindUint64,
	KindUintptr,
	KindFloat32,
	KindFloat64,
	KindComplex64,
	KindComplex128,
	KindArray,
	KindChan,
	KindFunc,
	KindInterface,
	KindMap,
	KindPtr,
	KindSlice,
	KindString,
	KindStruct,
	KindUnsafePointer,

	KindNoPointers = 1<<7,
};

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

static Sym*
typestruct(Type *t)
{
	// We use a weak reference to the reflect type
	// to avoid requiring package reflect in every binary.
	// If package reflect is available, the interface{} holding
	// a runtime type will contain a *reflect.commonType.
	// Otherwise it will use a nil type word but still be usable
	// by package runtime (because we always use the memory
	// after the interface value, not the interface value itself).
	return pkglookup("*reflect.commonType", weaktypepkg);
}

int
haspointers(Type *t)
{
	Type *t1;

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
	case TBOOL:
		return 0;
	case TARRAY:
		if(t->bound < 0)	// slice
			return 1;
		return haspointers(t->type);
	case TSTRUCT:
		for(t1=t->type; t1!=T; t1=t1->down)
			if(haspointers(t1->type))
				return 1;
		return 0;
	case TSTRING:
	case TPTR32:
	case TPTR64:
	case TUNSAFEPTR:
	case TINTER:
	case TCHAN:
	case TMAP:
	case TFUNC:
	default:
		return 1;
	}
}

/*
 * commonType
 * ../../pkg/runtime/type.go:/commonType
 */
static int
dcommontype(Sym *s, int ot, Type *t)
{
	int i, alg, sizeofAlg;
	Sym *sptr, *algsym;
	static Sym *algarray;
	char *p;

	sizeofAlg = 4*widthptr;
	if(algarray == nil)
		algarray = pkglookup("algarray", runtimepkg);
	alg = algtype(t);
	algsym = S;
	if(alg < 0)
		algsym = dalgsym(t);

	dowidth(t);
	if(t->sym != nil && !isptr[t->etype])
		sptr = dtypesym(ptrto(t));
	else
		sptr = weaktypesym(ptrto(t));

	// empty interface pointing at this type.
	// all the references that we emit are *interface{};
	// they point here.
	ot = rnd(ot, widthptr);
	ot = dsymptr(s, ot, typestruct(t), 0);
	ot = dsymptr(s, ot, s, 2*widthptr);

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

	i = kinds[t->etype];
	if(t->etype == TARRAY && t->bound < 0)
		i = KindSlice;
	if(!haspointers(t))
		i |= KindNoPointers;
	ot = duint8(s, ot, i);  // kind
	if(alg >= 0)
		ot = dsymptr(s, ot, algarray, alg*sizeofAlg);
	else
		ot = dsymptr(s, ot, algsym, 0);
	ot = duintptr(s, ot, 0);  // gc
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

Node*
typename(Type *t)
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

static Sym*
dtypesym(Type *t)
{
	int ot, xt, n, isddd, dupok;
	Sym *s, *s1, *s2;
	Sig *a, *m;
	Type *t1, *tbase, *t2;

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
	dupok = tbase->sym == S;

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
		xt = ot - 2*widthptr;
		break;

	case TARRAY:
		if(t->bound >= 0) {
			// ../../pkg/runtime/type.go:/ArrayType
			s1 = dtypesym(t->type);
			t2 = typ(TARRAY);
			t2->type = t->type;
			t2->bound = -1;  // slice
			s2 = dtypesym(t2);
			ot = dcommontype(s, ot, t);
			xt = ot - 2*widthptr;
			ot = dsymptr(s, ot, s1, 0);
			ot = dsymptr(s, ot, s2, 0);
			ot = duintptr(s, ot, t->bound);
		} else {
			// ../../pkg/runtime/type.go:/SliceType
			s1 = dtypesym(t->type);
			ot = dcommontype(s, ot, t);
			xt = ot - 2*widthptr;
			ot = dsymptr(s, ot, s1, 0);
		}
		break;

	case TCHAN:
		// ../../pkg/runtime/type.go:/ChanType
		s1 = dtypesym(t->type);
		ot = dcommontype(s, ot, t);
		xt = ot - 2*widthptr;
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
		xt = ot - 2*widthptr;
		ot = duint8(s, ot, isddd);

		// two slice headers: in and out.
		ot = rnd(ot, widthptr);
		ot = dsymptr(s, ot, s, ot+2*(widthptr+2*4));
		n = t->thistuple + t->intuple;
		ot = duint32(s, ot, n);
		ot = duint32(s, ot, n);
		ot = dsymptr(s, ot, s, ot+1*(widthptr+2*4)+n*widthptr);
		ot = duint32(s, ot, t->outtuple);
		ot = duint32(s, ot, t->outtuple);

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

		// ../../pkg/runtime/type.go:/InterfaceType
		ot = dcommontype(s, ot, t);
		xt = ot - 2*widthptr;
		ot = dsymptr(s, ot, s, ot+widthptr+2*4);
		ot = duint32(s, ot, n);
		ot = duint32(s, ot, n);
		for(a=m; a; a=a->link) {
			// ../../pkg/runtime/type.go:/imethod
			ot = dgostringptr(s, ot, a->name);
			ot = dgopkgpath(s, ot, a->pkg);
			ot = dsymptr(s, ot, dtypesym(a->type), 0);
		}
		break;

	case TMAP:
		// ../../pkg/runtime/type.go:/MapType
		s1 = dtypesym(t->down);
		s2 = dtypesym(t->type);
		ot = dcommontype(s, ot, t);
		xt = ot - 2*widthptr;
		ot = dsymptr(s, ot, s1, 0);
		ot = dsymptr(s, ot, s2, 0);
		break;

	case TPTR32:
	case TPTR64:
		if(t->type->etype == TANY) {
			// ../../pkg/runtime/type.go:/UnsafePointerType
			ot = dcommontype(s, ot, t);
			break;
		}
		// ../../pkg/runtime/type.go:/PtrType
		s1 = dtypesym(t->type);
		ot = dcommontype(s, ot, t);
		xt = ot - 2*widthptr;
		ot = dsymptr(s, ot, s1, 0);
		break;

	case TSTRUCT:
		// ../../pkg/runtime/type.go:/StructType
		// for security, only the exported fields.
		n = 0;
		for(t1=t->type; t1!=T; t1=t1->down) {
			dtypesym(t1->type);
			n++;
		}
		ot = dcommontype(s, ot, t);
		xt = ot - 2*widthptr;
		ot = dsymptr(s, ot, s, ot+widthptr+2*4);
		ot = duint32(s, ot, n);
		ot = duint32(s, ot, n);
		for(t1=t->type; t1!=T; t1=t1->down) {
			// ../../pkg/runtime/type.go:/structField
			if(t1->sym && !t1->embedded) {
				ot = dgostringptr(s, ot, t1->sym->name);
				if(exportname(t1->sym->name))
					ot = dgostringptr(s, ot, nil);
				else
					ot = dgopkgpath(s, ot, t1->sym->pkg);
			} else {
				ot = dgostringptr(s, ot, nil);
				ot = dgostringptr(s, ot, nil);
			}
			ot = dsymptr(s, ot, dtypesym(t1->type), 0);
			ot = dgostrlitptr(s, ot, t1->note);
			ot = duintptr(s, ot, t1->width);	// field offset
		}
		break;
	}
	ot = dextratype(s, ot, t, xt);
	ggloblsym(s, ot, dupok, 1);
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
		dimportpath(mkpkg(strlit("main")));
	}
}

static Sym*
dalgsym(Type *t)
{
	int ot;
	Sym *s, *hash, *eq;
	char buf[100];

	// dalgsym is only called for a type that needs an algorithm table,
	// which implies that the type is comparable (or else it would use ANOEQ).

	s = typesymprefix(".alg", t);
	hash = typesymprefix(".hash", t);
	genhash(hash, t);
	eq = typesymprefix(".eq", t);
	geneq(eq, t);

	// ../../pkg/runtime/runtime.h:/Alg
	ot = 0;
	ot = dsymptr(s, ot, hash, 0);
	ot = dsymptr(s, ot, eq, 0);
	ot = dsymptr(s, ot, pkglookup("memprint", runtimepkg), 0);
	switch(t->width) {
	default:
		ot = dsymptr(s, ot, pkglookup("memcopy", runtimepkg), 0);
		break;
	case 1:
	case 2:
	case 4:
	case 8:
	case 16:
		snprint(buf, sizeof buf, "memcopy%d", (int)t->width*8);
		ot = dsymptr(s, ot, pkglookup(buf, runtimepkg), 0);
		break;
	}

	ggloblsym(s, ot, 1, 1);
	return s;
}


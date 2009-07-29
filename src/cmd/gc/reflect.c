// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go.h"

/*
 * runtime interface and reflection data structures
 */

static Sym* dtypesym(Type*);

static int
sigcmp(Sig *a, Sig *b)
{
	return strcmp(a->name, b->name);
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
 * return function type, receiver as first argument.
 */
static Type*
methodfunc(Type *f)
{
	NodeList *in, *out;
	Node *d;
	Type *t;

	in = nil;
	if(!isifacemethod(f)) {
		d = nod(ODCLFIELD, N, N);
		d->type = getthisx(f->type)->type->type;
		in = list(in, d);
	}
	for(t=getinargx(f->type)->type; t; t=t->down) {
		d = nod(ODCLFIELD, N, N);
		d->type = t->type;
		in = list(in, d);
	}

	out = nil;
	for(t=getoutargx(f->type)->type; t; t=t->down) {
		d = nod(ODCLFIELD, N, N);
		d->type = t->type;
		out = list(out, d);
	}

	return functype(N, in, out);
}

/*
 * return methods of non-interface type t,
 * sorted by hash.
 * generates stub functions as needed.
 */
static Sig*
methods(Type *t)
{
	int o;
	Type *f, *mt, *it, *this;
	Sig *a, *b;
	Sym *method;
	Prog *oldlist;

	// named method type
	mt = methtype(t);
	if(mt == T)
		return nil;
	expandmeth(mt->sym, mt);

	// type stored in interface word
	it = t;
	if(it->width > widthptr)
		it = ptrto(t);

	// make list of methods for t,
	// generating code if necessary.
	a = nil;
	o = 0;
	oldlist = nil;
	for(f=mt->xmethod; f; f=f->down) {
		if(f->type->etype != TFUNC)
			continue;
		if(f->etype != TFIELD)
			fatal("methods: not field");
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
		&& f->embedded != 2 && !isifacemethod(f))
			continue;

		b = mal(sizeof(*b));
		b->link = a;
		a = b;

		a->name = method->name;
		a->hash = PRIME8*stringhash(a->name) + PRIME9*typehash(f->type, 0, 0);
		if(!exportname(a->name)) {
			a->package = method->package;
			a->hash += PRIME10*stringhash(a->package);
		}
		a->perm = o++;
		a->isym = methodsym(method, it);
		a->tsym = methodsym(method, t);
		a->type = methodfunc(f);

		if(!a->isym->siggen) {
			a->isym->siggen = 1;
			if(!eqtype(this, it)) {
				if(oldlist == nil)
					oldlist = pc;
				// Is okay to call genwrapper here always,
				// but we can generate more efficient code
				// using genembedtramp if all that is necessary
				// is a pointer adjustment and a JMP.
				if(isptr[it->etype] && isptr[this->etype]
				&& f->embedded && !isifacemethod(f))
					genembedtramp(it, f, a->isym);
				else
					genwrapper(it, f, a->isym);
			}
		}

		if(!a->tsym->siggen) {
			a->tsym->siggen = 1;
			if(!eqtype(this, t)) {
				if(oldlist == nil)
					oldlist = pc;
				if(isptr[t->etype] && isptr[this->etype]
				&& f->embedded && !isifacemethod(f))
					genembedtramp(t, f, a->tsym);
				else
					genwrapper(t, f, a->tsym);
			}
		}
	}

	// restore data output
	if(oldlist) {
		// old list ended with AEND; change to ANOP
		// so that the trampolines that follow can be found.
		nopout(oldlist);

		// start new data list
		newplist();
	}

	return lsort(a, sigcmp);
}

/*
 * return methods of interface type t, sorted by hash.
 */
Sig*
imethods(Type *t)
{
	Sig *a, *b;
	int o;
	Type *f;

	a = nil;
	o = 0;
	for(f=t->type; f; f=f->down) {
		if(f->etype != TFIELD)
			fatal("imethods: not field");
		if(f->type->etype != TFUNC || f->sym == nil)
			continue;
		b = mal(sizeof(*b));
		b->link = a;
		a = b;

		a->name = f->sym->name;
		a->hash = PRIME8*stringhash(a->name) + PRIME9*typehash(f->type, 0, 0);
		if(!exportname(a->name)) {
			a->package = f->sym->package;
			a->hash += PRIME10*stringhash(a->package);
		}
		a->perm = o++;
		a->offset = 0;
		a->type = methodfunc(f);
	}

	return lsort(a, sigcmp);
}

/*
 * uncommonType
 * ../../pkg/runtime/type.go:/uncommonType
 */
static Sym*
dextratype(Type *t)
{
	int ot, n;
	char *p;
	Sym *s;
	Sig *a, *m;

	m = methods(t);
	if(t->sym == nil && m == nil)
		return nil;

	n = 0;
	for(a=m; a; a=a->link) {
		dtypesym(a->type);
		n++;
	}

	p = smprint("%#-T", t);
	s = pkglookup(p, "extratype");
	ot = 0;
	if(t->sym) {
		ot = dgostringptr(s, ot, t->sym->name);
		if(t != types[t->etype])
			ot = dgostringptr(s, ot, t->sym->package);
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
		ot = duint32(s, ot, a->hash);
		ot = rnd(ot, widthptr);
		ot = dgostringptr(s, ot, a->name);
		ot = dgostringptr(s, ot, a->package);
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
	ggloblsym(s, ot, 1);

	return s;
}

static char*
structnames[] =
{
	[TINT]		= "*runtime.IntType",
	[TUINT]		= "*runtime.UintType",
	[TINT8]		= "*runtime.Int8Type",
	[TUINT8]	= "*runtime.Uint8Type",
	[TINT16]	= "*runtime.Int16Type",
	[TUINT16]	= "*runtime.Uint16Type",
	[TINT32]	= "*runtime.Int32Type",
	[TUINT32]	= "*runtime.Uint32Type",
	[TINT64]	= "*runtime.Int64Type",
	[TUINT64]	= "*runtime.Uint64Type",
	[TUINTPTR]	= "*runtime.UintptrType",
	[TFLOAT]	= "*runtime.FloatType",
	[TFLOAT32]	= "*runtime.Float32Type",
	[TFLOAT64]	= "*runtime.Float64Type",
	[TBOOL]		= "*runtime.BoolType",
	[TSTRING]		= "*runtime.StringType",
	[TDDD]		= "*runtime.DotDotDotType",

	[TPTR32]		= "*runtime.PtrType",
	[TPTR64]		= "*runtime.PtrType",
	[TSTRUCT]	= "*runtime.StructType",
	[TINTER]		= "*runtime.InterfaceType",
	[TCHAN]		= "*runtime.ChanType",
	[TMAP]		= "*runtime.MapType",
	[TARRAY]		= "*runtime.ArrayType",
	[TFUNC]		= "*runtime.FuncType",
};

static Sym*
typestruct(Type *t)
{
	char *name;
	int et;

	et = t->etype;
	if(et < 0 || et >= nelem(structnames) || (name = structnames[et]) == nil) {
		fatal("typestruct %lT", t);
		return nil;	// silence gcc
	}

	if(isslice(t))
		name = "*runtime.SliceType";

	if(isptr[et] && t->type->etype == TANY)
		name = "*runtime.UnsafePointerType";

	return pkglookup(name, "type");
}

/*
 * commonType
 * ../../pkg/runtime/type.go:/commonType
 */
static int
dcommontype(Sym *s, int ot, Type *t)
{
	int i;
	Sym *s1;
	Type *elem;
	char *p;

	s1 = dextratype(t);

	// empty interface pointing at this type.
	// all the references that we emit are *interface{};
	// they point here.
	ot = rnd(ot, widthptr);
	ot = dsymptr(s, ot, typestruct(t), 0);
	ot = dsymptr(s, ot, s, 2*widthptr);

	// ../../pkg/runtime/type.go:/commonType
	// actual type structure
	//	type commonType struct {
	//		size uintptr;
	//		hash uint32;
	//		alg uint8;
	//		align uint8;
	//		fieldAlign uint8;
	//		string *string;
	//		*nameInfo;
	//	}
	ot = duintptr(s, ot, t->width);
	ot = duint32(s, ot, typehash(t, 1, 0));
	ot = duint8(s, ot, algtype(t));
	elem = t;
	while(elem->etype == TARRAY && elem->bound >= 0)
		elem = elem->type;
	i = elem->width;
	if(i > maxround)
		i = maxround;
	ot = duint8(s, ot, i);	// align
	ot = duint8(s, ot, i);	// fieldAlign
	p = smprint("%#-T", t);
	ot = dgostringptr(s, ot, p);	// string
	free(p);
	if(s1)
		ot = dsymptr(s, ot, s1, 0);	// extraType
	else
		ot = duintptr(s, ot, 0);

	return ot;
}

Sym*
typesym(Type *t)
{
	char *p;
	Sym *s;

	p = smprint("%#-T", t);
	s = pkglookup(p, "type");
	free(p);
	return s;
}

Node*
typename(Type *t)
{
	Sym *s;
	Node *n;
	Dcl *d;

	s = typesym(t);
	if(s->def == N) {
		n = nod(ONAME, N, N);
		n->sym = s;
		n->type = types[TUINT8];
		n->addable = 1;
		n->ullman = 1;
		n->class = PEXTERN;
		n->xoffset = 0;
		s->def = n;

		// copy to signatlist
		d = dcl();
		d->dsym = s;
		d->dtype = t;
		d->op = OTYPE;
		d->forw = signatlist;
		signatlist = d;
	}

	n = nod(OADDR, s->def, N);
	n->type = ptrto(s->def->type);
	n->addable = 1;
	n->ullman = 2;
	return n;
}

Sym*
dtypesym(Type *t)
{
	int ot, n;
	Sym *s, *s1, *s2;
	Sig *a, *m;
	Type *t1;

	s = typesym(t);
	if(s->siggen)
		return s;
	s->siggen = 1;

	// special case (look for runtime below):
	// when compiling package runtime,
	// emit the type structures for int, float, etc.
	t1 = T;
	if(isptr[t->etype])
		t1 = t->type;

	if(strcmp(package, "runtime") == 0) {
		if(t == types[t->etype])
			goto ok;
		if(t1 && t1 == types[t1->etype])
			goto ok;
	}

	// named types from other files are defined in those files
	if(t->sym && !t->local)
		return s;
	if(!t->sym && t1 && t1->sym && !t1->local)
		return s;
	if(isforw[t->etype] || (t1 && isforw[t1->etype]))
		return s;

ok:
	ot = 0;
	switch(t->etype) {
	default:
		ot = dcommontype(s, ot, t);
		break;

	case TARRAY:
		// ../../pkg/runtime/type.go:/ArrayType
		s1 = dtypesym(t->type);
		ot = dcommontype(s, ot, t);
		ot = dsymptr(s, ot, s1, 0);
		if(t->bound < 0)
			ot = duintptr(s, ot, -1);
		else
			ot = duintptr(s, ot, t->bound);
		break;

	case TCHAN:
		// ../../pkg/runtime/type.go:/ChanType
		s1 = dtypesym(t->type);
		ot = dcommontype(s, ot, t);
		ot = dsymptr(s, ot, s1, 0);
		ot = duintptr(s, ot, t->chan);
		break;

	case TFORWSTRUCT:
	case TFORWINTER:
		return s;

	case TFUNC:
		for(t1=getthisx(t)->type; t1; t1=t1->down)
			dtypesym(t1->type);
		for(t1=getinargx(t)->type; t1; t1=t1->down)
			dtypesym(t1->type);
		for(t1=getoutargx(t)->type; t1; t1=t1->down)
			dtypesym(t1->type);

		ot = dcommontype(s, ot, t);

		// two slice headers: in and out.
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
		ot = dsymptr(s, ot, s, ot+widthptr+2*4);
		ot = duint32(s, ot, n);
		ot = duint32(s, ot, n);
		for(a=m; a; a=a->link) {
			// ../../pkg/runtime/type.go:/imethod
			ot = duint32(s, ot, a->hash);
			ot = duint32(s, ot, a->perm);
			ot = dgostringptr(s, ot, a->name);
			ot = dgostringptr(s, ot, a->package);
			ot = dsymptr(s, ot, dtypesym(a->type), 0);
		}
		break;

	case TMAP:
		// ../../pkg/runtime/type.go:/MapType
		s1 = dtypesym(t->down);
		s2 = dtypesym(t->type);
		ot = dcommontype(s, ot, t);
		ot = dsymptr(s, ot, s1, 0);
		ot = dsymptr(s, ot, s2, 0);
		break;

	case TPTR32:
	case TPTR64:
		if(t->type->etype == TANY) {
			ot = dcommontype(s, ot, t);
			break;
		}
		// ../../pkg/runtime/type.go:/PtrType
		s1 = dtypesym(t->type);
		ot = dcommontype(s, ot, t);
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
					ot = dgostringptr(s, ot, t1->sym->package);
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

	ggloblsym(s, ot, 1);
	return s;
}

void
dumptypestructs(void)
{
	int i;
	Dcl *d, *x;
	Type *t;

	// copy types from externdcl list to signatlist
	for(d=externdcl; d!=D; d=d->forw) {
		if(d->op != OTYPE)
			continue;
		t = d->dtype;
		x = mal(sizeof(*x));
		x->op = OTYPE;
		x->dtype = t;
		x->forw = signatlist;
		x->block = 0;
		signatlist = x;
	}

	// process signatlist
	for(d=signatlist; d!=D; d=d->forw) {
		if(d->op != OTYPE)
			continue;
		t = d->dtype;
		dtypesym(t);
		if(t->sym && !isptr[t->etype])
			dtypesym(ptrto(t));
	}

	// do basic types if compiling package runtime, type.go.
	// they have to be in at least one package,
	// and reflect is always loaded implicitly,
	// so this is as good as any.
	// another possible choice would be package main,
	// but using runtime means fewer copies in .6 files.
	if(strcmp(package, "runtime") == 0 && strcmp(filename, "type") == 0) {
		for(i=1; i<=TBOOL; i++)
			if(i != TFLOAT80)
				dtypesym(ptrto(types[i]));
		dtypesym(ptrto(types[TSTRING]));
		dtypesym(typ(TDDD));
		dtypesym(ptrto(pkglookup("Pointer", "unsafe")->def->type));
	}
}

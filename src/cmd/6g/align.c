// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "gg.h"

int
main(int argc, char *argv[])
{
	mainlex(argc, argv);
	return 99;
}

/*
 * machine size and rounding
 * alignment is dictated around
 * the size of a pointer.
 * the size of the generic types
 * are pulled from the typedef table.
 */

static	int	wptr	= 8;	// width of a pointer
static	int	wmax	= 8;	// max rounding

uint32
rnd(uint32 o, uint32 r)
{
	if(r > wmax)
		r = wmax;
	if(r != 0)
		while(o%r != 0)
			o++;
	return o;
}

void
offmod(Type *t)
{
	Type *f;
	int32 o;

	o = 0;
	for(f=t->type; f!=T; f=f->down) {
		if(f->etype != TFIELD)
			fatal("widstruct: not TFIELD: %lT", f);
		if(f->type->etype != TFUNC)
			continue;
		f->width = o;
		o += wptr;
	}
}

uint32
widstruct(Type *t, uint32 o, int flag)
{
	Type *f;
	int32 w;

	for(f=t->type; f!=T; f=f->down) {
		if(f->etype != TFIELD)
			fatal("widstruct: not TFIELD: %lT", f);
		dowidth(f->type);
		w = f->type->width;
		o = rnd(o, w);
		f->width = o;	// really offset for TFIELD
		o += w;
	}
	// final width is rounded
	if(flag)
		o = rnd(o, maxround);
	t->width = o;
	return o;
}

void
dowidth(Type *t)
{
	uint32 w;

	if(t == T)
		return;

	if(t->width == -2) {
		yyerror("invalid recursive type %T", t);
		t->width = 0;
		return;
	}

	t->width = -2;

	w = 0;
	switch(simtype[t->etype]) {
	default:
		fatal("dowidth: unknown type: %E", t->etype);
		break;

	/* compiler-specific stuff */
	case TINT8:
	case TUINT8:
	case TBOOL:		// bool is int8
		w = 1;
		break;
	case TINT16:
	case TUINT16:
		w = 2;
		break;
	case TINT32:
	case TUINT32:
	case TFLOAT32:
	case TPTR32:		// note lack of recursion
		w = 4;
		break;
	case TINT64:
	case TUINT64:
	case TFLOAT64:
	case TPTR64:		// note lack of recursion
		w = 8;
		break;
	case TFLOAT80:
		w = 10;
		break;
	case TDDD:
		w = 2*wptr;
		break;
	case TINTER:		// implemented as 2 pointers
	case TFORWINTER:
		offmod(t);
		w = 2*wptr;
		break;
	case TCHAN:		// implemented as pointer
		dowidth(t->type);
		dowidth(t->down);
		w = wptr;
		break;
	case TMAP:		// implemented as pointer
		dowidth(t->type);
		w = wptr;
		break;
	case TFORW:		// should have been filled in
	case TFORWSTRUCT:
		yyerror("incomplete type %T", t);
		w = wptr;
		break;
	case TANY:		// implemented as pointer
		w = wptr;
		break;
	case TSTRING:		// implemented as pointer
		w = wptr;
		break;
	case TARRAY:
		dowidth(t->type);
		if(t->bound >= 0 && t->type != T)
			w = t->bound * t->type->width;
		break;

	case TSTRUCT:
		if(t->funarg)
			fatal("dowidth fn struct %T", t);
		w = widstruct(t, 0, 1);
		offmod(t);
		break;

	case TFUNC:
		// function is 3 cated structures
		w = widstruct(*getthis(t), 0, 1);
		w = widstruct(*getinarg(t), w, 0);
		w = widstruct(*getoutarg(t), w, 1);
		t->argwid = w;
		w = 0;
		break;
	}
	t->width = w;
}

void
besetptr(void)
{
	maxround = wmax;
	widthptr = wptr;

	types[TPTR32] = typ(TPTR32);
	dowidth(types[TPTR32]);

	types[TPTR64] = typ(TPTR64);
	dowidth(types[TPTR64]);

	tptr = TPTR32;
	if(wptr == 8)
		tptr = TPTR64;
}

/*
 * additionally, go declares several platform-specific type aliases:
 * int, uint, float, and uptrint
 */
static	struct
{
	char*	name;
	int	etype;
	int	sameas;
}
typedefs[] =
{
	"int",		TINT,		TINT32,
	"uint",		TUINT,		TUINT32,
	"uintptr",	TUINTPTR,	TUINT64,
	"float",	TFLOAT,		TFLOAT32,
};

void
belexinit(int lextype)
{
	int i, etype, sameas;
	Sym *s;
	Type *t;

	zprog.link = P;
	zprog.as = AGOK;
	zprog.from.type = D_NONE;
	zprog.from.index = D_NONE;
	zprog.from.scale = 0;
	zprog.to = zprog.from;

	for(i=0; i<nelem(typedefs); i++) {
		s = lookup(typedefs[i].name);
		s->lexical = lextype;

		etype = typedefs[i].etype;
		if(etype < 0 || etype >= nelem(types))
			fatal("lexinit: %s bad etype", s->name);
		sameas = typedefs[i].sameas;
		if(sameas < 0 || sameas >= nelem(types))
			fatal("lexinit: %s bad sameas", s->name);
		simtype[etype] = sameas;

		t = types[etype];
		if(t != T)
			fatal("lexinit: %s already defined", s->name);

		t = typ(etype);
		t->sym = s;

		dowidth(t);
		types[etype] = t;
		s->otype = t;

		if(minfltval[sameas] != nil)
			minfltval[etype] = minfltval[sameas];
		if(maxfltval[sameas] != nil)
			maxfltval[etype] = maxfltval[sameas];
		if(minintval[sameas] != nil)
			minintval[etype] = minintval[sameas];
		if(maxintval[sameas] != nil)
			maxintval[etype] = maxintval[sameas];
	}

	symstringo = lookup(".stringo");	// strings

	listinit();
}

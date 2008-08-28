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

/*
 * additionally, go declares several platform-specific type aliases:
 * ushort, short, uint, int, uint32, int32, float, and double.  The bit
 */
static char*
typedefs[] =
{
	"short",	"int16",	// shorts
	"ushort",	"uint16",

	"int",		"int32",	// ints
	"uint",		"uint32",
//	"rune",		"uint32",

	"long",		"int64",	// longs
	"ulong",	"uint64",

//	"vlong",	"int64",	// vlongs
//	"uvlong",	"uint64",

	"float",	"float32",	// floats
	"double",	"float64",

};

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

	w = 0;
	if(t == T)
		return;

	switch(t->etype) {
	default:
		fatal("dowidth: unknown type: %E", t->etype);
		break;

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
	case TPTR32:
		w = 4;
		break;
	case TINT64:
	case TUINT64:
	case TFLOAT64:
	case TPTR64:
		w = 8;
		break;
	case TFLOAT80:
		w = 10;
		break;
	case TINTER:		// implemented as 2 pointers
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
	case TFORW:		// implemented as pointer
		w = wptr;
		break;
	case TANY:		// implemented as pointer
		w = wptr;
		break;
	case TSTRING:		// implemented as pointer
		w = wptr;
		break;
	case TDARRAY:
		fatal("width of a dynamic array");
	case TARRAY:
		if(t->type == T)
			break;
		dowidth(t->type);
		w = t->bound * t->type->width
;//			+ offsetof(Array, b[0]);
		break;

	case TSTRUCT:
		w = widstruct(t, 0, 1);
		offmod(t);
		break;

	case TFUNC:
		// function is 3 cated structures
		w = widstruct(*getthis(t), 0, 0);
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

void
belexinit(int lextype)
{
	int i;
	Sym *s0, *s1;

	zprog.link = P;
	zprog.as = AGOK;
	zprog.from.type = D_NONE;
	zprog.from.index = D_NONE;
	zprog.from.scale = 0;
	zprog.to = zprog.from;

	for(i=0; i<nelem(typedefs); i+=2) {
		s1 = lookup(typedefs[i+1]);
		if(s1->lexical != lextype)
			yyerror("need %s to define %s",
				typedefs[i+1], typedefs[i+0]);
		s0 = lookup(typedefs[i+0]);
		s0->lexical = s1->lexical;
		s0->otype = s1->otype;
	}

	symstringo = lookup(".stringo");	// strings

	listinit();
	buildtxt();
}

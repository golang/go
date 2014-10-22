// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	<u.h>
#include	<libc.h>
#include	"go.h"
#include	"opnames.h"

//
// Format conversions
//	%L int		Line numbers
//
//	%E int		etype values (aka 'Kind')
//
//	%O int		Node Opcodes
//		Flags: "%#O": print go syntax. (automatic unless fmtmode == FDbg)
//
//	%J Node*	Node details
//		Flags: "%hJ" suppresses things not relevant until walk.
//
//	%V Val*		Constant values
//
//	%S Sym*		Symbols
//		Flags: +,- #: mode (see below)
//			"%hS"	unqualified identifier in any mode
//			"%hhS"  in export mode: unqualified identifier if exported, qualified if not
//
//	%T Type*	Types
//		Flags: +,- #: mode (see below)
//			'l' definition instead of name.
//			'h' omit "func" and receiver in function types
//			'u' (only in -/Sym mode) print type identifiers wit package name instead of prefix.
//
//	%N Node*	Nodes
//		Flags: +,- #: mode (see below)
//			'h' (only in +/debug mode) suppress recursion
//			'l' (only in Error mode) print "foo (type Bar)"
//
//	%H NodeList*	NodeLists
//		Flags: those of %N
//			','  separate items with ',' instead of ';'
//
//	%Z Strlit*	String literals
//
//   In mparith1.c:
//      %B Mpint*	Big integers
//	%F Mpflt*	Big floats
//
//   %S, %T and %N obey use the following flags to set the format mode:
enum {
	FErr,	//     error mode (default)
	FDbg,	//     "%+N" debug mode
	FExp,	//     "%#N" export mode
	FTypeId,  //   "%-N" turning-types-into-symbols-mode: identical types give identical strings
};
static int fmtmode;
static int fmtpkgpfx;	// %uT stickyness
//
// E.g. for %S:	%+S %#S %-S	print an identifier properly qualified for debug/export/internal mode.
//
// The mode flags  +, - and # are sticky, meaning they persist through
// recursions of %N, %T and %S, but not the h and l flags.  The u flag is
// sticky only on %T recursions and only used in %-/Sym mode.

//
// Useful format combinations:
//
//	%+N   %+H	multiline recursive debug dump of node/nodelist
//	%+hN  %+hH	non recursive debug dump
//
//	%#N   %#T	export format
//	%#lT		type definition instead of name
//	%#hT		omit"func" and receiver in function signature
//
//	%lN		"foo (type Bar)" for error messages
//
//	%-T		type identifiers
//	%-hT		type identifiers without "func" and arg names in type signatures (methodsym)
//	%-uT		type identifiers with package name instead of prefix (typesym, dcommontype, typehash)
//


static int
setfmode(unsigned long *flags)
{
	int fm;

	fm = fmtmode;
	if(*flags & FmtSign)
		fmtmode = FDbg;
	else if(*flags & FmtSharp)
		fmtmode = FExp;
	else if(*flags & FmtLeft)
		fmtmode = FTypeId;

	*flags &= ~(FmtSharp|FmtLeft|FmtSign);
	return fm;
}

// Fmt "%L": Linenumbers
static int
Lconv(Fmt *fp)
{
	return linklinefmt(ctxt, fp);
}

static char*
goopnames[] =
{
	[OADDR]		= "&",
	[OADD]		= "+",
	[OADDSTR]	= "+",
	[OANDAND]	= "&&",
	[OANDNOT]	= "&^",
	[OAND]		= "&",
	[OAPPEND]	= "append",
	[OAS]		= "=",
	[OAS2]		= "=",
	[OBREAK]	= "break",
	[OCALL]		= "function call",	// not actual syntax
	[OCAP]		= "cap",
	[OCASE]		= "case",
	[OCLOSE]	= "close",
	[OCOMPLEX]	= "complex",
	[OCOM]		= "^",
	[OCONTINUE]	= "continue",
	[OCOPY]		= "copy",
	[ODEC]		= "--",
	[ODELETE]	= "delete",
	[ODEFER]	= "defer",
	[ODIV]		= "/",
	[OEQ]		= "==",
	[OFALL]		= "fallthrough",
	[OFOR]		= "for",
	[OGE]		= ">=",
	[OGOTO]		= "goto",
	[OGT]		= ">",
	[OIF]		= "if",
	[OIMAG]		= "imag",
	[OINC]		= "++",
	[OIND]		= "*",
	[OLEN]		= "len",
	[OLE]		= "<=",
	[OLSH]		= "<<",
	[OLT]		= "<",
	[OMAKE]		= "make",
	[OMINUS]	= "-",
	[OMOD]		= "%",
	[OMUL]		= "*",
	[ONEW]		= "new",
	[ONE]		= "!=",
	[ONOT]		= "!",
	[OOROR]		= "||",
	[OOR]		= "|",
	[OPANIC]	= "panic",
	[OPLUS]		= "+",
	[OPRINTN]	= "println",
	[OPRINT]	= "print",
	[ORANGE]	= "range",
	[OREAL]		= "real",
	[ORECV]		= "<-",
	[ORECOVER]	= "recover",
	[ORETURN]	= "return",
	[ORSH]		= ">>",
	[OSELECT]	= "select",
	[OSEND]		= "<-",
	[OSUB]		= "-",
	[OSWITCH]	= "switch",
	[OXOR]		= "^",
};

// Fmt "%O":  Node opcodes
static int
Oconv(Fmt *fp)
{
	int o;

	o = va_arg(fp->args, int);
	if((fp->flags & FmtSharp) || fmtmode != FDbg)
		if(o >= 0 && o < nelem(goopnames) && goopnames[o] != nil)
			return fmtstrcpy(fp, goopnames[o]);

	if(o >= 0 && o < nelem(opnames) && opnames[o] != nil)
		return fmtstrcpy(fp, opnames[o]);

	return fmtprint(fp, "O-%d", o);
}

static const char* classnames[] = {
	"Pxxx",
	"PEXTERN",
	"PAUTO",
	"PPARAM",
	"PPARAMOUT",
	"PPARAMREF",
	"PFUNC",
};

// Fmt "%J": Node details.
static int
Jconv(Fmt *fp)
{
	Node *n;
	char *s;
	int c;

	n = va_arg(fp->args, Node*);

	c = fp->flags&FmtShort;

	if(!c && n->ullman != 0)
		fmtprint(fp, " u(%d)", n->ullman);

	if(!c && n->addable != 0)
		fmtprint(fp, " a(%d)", n->addable);

	if(!c && n->vargen != 0)
		fmtprint(fp, " g(%d)", n->vargen);

	if(n->lineno != 0)
		fmtprint(fp, " l(%d)", n->lineno);

	if(!c && n->xoffset != BADWIDTH)
		fmtprint(fp, " x(%lld%+lld)", n->xoffset, n->stkdelta);

	if(n->class != 0) {
		s = "";
		if(n->class & PHEAP) s = ",heap";
		if((n->class & ~PHEAP) < nelem(classnames))
			fmtprint(fp, " class(%s%s)", classnames[n->class&~PHEAP], s);
		else
			fmtprint(fp, " class(%d?%s)", n->class&~PHEAP, s);
	}

	if(n->colas != 0)
		fmtprint(fp, " colas(%d)", n->colas);

	if(n->funcdepth != 0)
		fmtprint(fp, " f(%d)", n->funcdepth);

	switch(n->esc) {
	case EscUnknown:
		break;
	case EscHeap:
		fmtprint(fp, " esc(h)");
		break;
	case EscScope:
		fmtprint(fp, " esc(s)");
		break;
	case EscNone:
		fmtprint(fp, " esc(no)");
		break;
	case EscNever:
		if(!c)
			fmtprint(fp, " esc(N)");
		break;
	default:
		fmtprint(fp, " esc(%d)", n->esc);
		break;
	}

	if(n->escloopdepth)
		fmtprint(fp, " ld(%d)", n->escloopdepth);

	if(!c && n->typecheck != 0)
		fmtprint(fp, " tc(%d)", n->typecheck);

	if(!c && n->dodata != 0)
		fmtprint(fp, " dd(%d)", n->dodata);

	if(n->isddd != 0)
		fmtprint(fp, " isddd(%d)", n->isddd);

	if(n->implicit != 0)
		fmtprint(fp, " implicit(%d)", n->implicit);

	if(n->embedded != 0)
		fmtprint(fp, " embedded(%d)", n->embedded);

	if(!c && n->used != 0)
		fmtprint(fp, " used(%d)", n->used);
	return 0;
}

// Fmt "%V": Values
static int
Vconv(Fmt *fp)
{
	Val *v;
	vlong x;

	v = va_arg(fp->args, Val*);

	switch(v->ctype) {
	case CTINT:
		if((fp->flags & FmtSharp) || fmtmode == FExp)
			return fmtprint(fp, "%#B", v->u.xval);
		return fmtprint(fp, "%B", v->u.xval);
	case CTRUNE:
		x = mpgetfix(v->u.xval);
		if(' ' <= x && x < 0x80 && x != '\\' && x != '\'')
			return fmtprint(fp, "'%c'", (int)x);
		if(0 <= x && x < (1<<16))
			return fmtprint(fp, "'\\u%04ux'", (int)x);
		if(0 <= x && x <= Runemax)
			return fmtprint(fp, "'\\U%08llux'", x);
		return fmtprint(fp, "('\\x00' + %B)", v->u.xval);
	case CTFLT:
		if((fp->flags & FmtSharp) || fmtmode == FExp)
			return fmtprint(fp, "%F", v->u.fval);
		return fmtprint(fp, "%#F", v->u.fval);
	case CTCPLX:
		if((fp->flags & FmtSharp) || fmtmode == FExp)
			return fmtprint(fp, "(%F+%Fi)", &v->u.cval->real, &v->u.cval->imag);
		if(mpcmpfltc(&v->u.cval->real, 0) == 0)
			return fmtprint(fp, "%#Fi", &v->u.cval->imag);
		if(mpcmpfltc(&v->u.cval->imag, 0) == 0)
			return fmtprint(fp, "%#F", &v->u.cval->real);
		if(mpcmpfltc(&v->u.cval->imag, 0) < 0)
			return fmtprint(fp, "(%#F%#Fi)", &v->u.cval->real, &v->u.cval->imag);
		return fmtprint(fp, "(%#F+%#Fi)", &v->u.cval->real, &v->u.cval->imag);
	case CTSTR:
		return fmtprint(fp, "\"%Z\"", v->u.sval);
	case CTBOOL:
		if( v->u.bval)
			return fmtstrcpy(fp, "true");
		return fmtstrcpy(fp, "false");
	case CTNIL:
		return fmtstrcpy(fp, "nil");
	}
	return fmtprint(fp, "<ctype=%d>", v->ctype);
}

// Fmt "%Z": escaped string literals
static int
Zconv(Fmt *fp)
{
	Rune r;
	Strlit *sp;
	char *s, *se;
	int n;

	sp = va_arg(fp->args, Strlit*);
	if(sp == nil)
		return fmtstrcpy(fp, "<nil>");

	s = sp->s;
	se = s + sp->len;

	// NOTE: Keep in sync with ../ld/go.c:/^Zconv.
	while(s < se) {
		n = chartorune(&r, s);
		s += n;
		switch(r) {
		case Runeerror:
			if(n == 1) {
				fmtprint(fp, "\\x%02x", (uchar)*(s-1));
				break;
			}
			// fall through
		default:
			if(r < ' ') {
				fmtprint(fp, "\\x%02x", r);
				break;
			}
			fmtrune(fp, r);
			break;
		case '\t':
			fmtstrcpy(fp, "\\t");
			break;
		case '\n':
			fmtstrcpy(fp, "\\n");
			break;
		case '\"':
		case '\\':
			fmtrune(fp, '\\');
			fmtrune(fp, r);
			break;
		case 0xFEFF: // BOM, basically disallowed in source code
			fmtstrcpy(fp, "\\uFEFF");
			break;
		}
	}
	return 0;
}

/*
s%,%,\n%g
s%\n+%\n%g
s%^[	]*T%%g
s%,.*%%g
s%.+%	[T&]		= "&",%g
s%^	........*\]%&~%g
s%~	%%g
*/

static char*
etnames[] =
{
	[TINT]		= "INT",
	[TUINT]		= "UINT",
	[TINT8]		= "INT8",
	[TUINT8]	= "UINT8",
	[TINT16]	= "INT16",
	[TUINT16]	= "UINT16",
	[TINT32]	= "INT32",
	[TUINT32]	= "UINT32",
	[TINT64]	= "INT64",
	[TUINT64]	= "UINT64",
	[TUINTPTR]	= "UINTPTR",
	[TFLOAT32]	= "FLOAT32",
	[TFLOAT64]	= "FLOAT64",
	[TCOMPLEX64]	= "COMPLEX64",
	[TCOMPLEX128]	= "COMPLEX128",
	[TBOOL]		= "BOOL",
	[TPTR32]	= "PTR32",
	[TPTR64]	= "PTR64",
	[TFUNC]		= "FUNC",
	[TARRAY]	= "ARRAY",
	[TSTRUCT]	= "STRUCT",
	[TCHAN]		= "CHAN",
	[TMAP]		= "MAP",
	[TINTER]	= "INTER",
	[TFORW]		= "FORW",
	[TFIELD]	= "FIELD",
	[TSTRING]	= "STRING",
	[TANY]		= "ANY",
};

// Fmt "%E": etype
static int
Econv(Fmt *fp)
{
	int et;

	et = va_arg(fp->args, int);
	if(et >= 0 && et < nelem(etnames) && etnames[et] != nil)
		return fmtstrcpy(fp, etnames[et]);
	return fmtprint(fp, "E-%d", et);
}

// Fmt "%S": syms
static int
symfmt(Fmt *fp, Sym *s)
{
	char *p;

	if(s->pkg && !(fp->flags&FmtShort)) {
		switch(fmtmode) {
		case FErr:	// This is for the user
			if(s->pkg == localpkg)
				return fmtstrcpy(fp, s->name);
			// If the name was used by multiple packages, display the full path,
			if(s->pkg->name && pkglookup(s->pkg->name, nil)->npkg > 1)
				return fmtprint(fp, "\"%Z\".%s", s->pkg->path, s->name);
			return fmtprint(fp, "%s.%s", s->pkg->name, s->name);
		case FDbg:
			return fmtprint(fp, "%s.%s", s->pkg->name, s->name);
		case FTypeId:
			if(fp->flags&FmtUnsigned)
				return fmtprint(fp, "%s.%s", s->pkg->name, s->name);	// dcommontype, typehash
			return fmtprint(fp, "%s.%s", s->pkg->prefix, s->name);	// (methodsym), typesym, weaksym
		case FExp:
			if(s->name && s->name[0] == '.')
				fatal("exporting synthetic symbol %s", s->name);
			if(s->pkg != builtinpkg)
				return fmtprint(fp, "@\"%Z\".%s", s->pkg->path, s->name);
		}
	}

	if(fp->flags&FmtByte) {  // FmtByte (hh) implies FmtShort (h)
		// skip leading "type." in method name
		p = utfrrune(s->name, '.');
		if(p)
			p++;
		else
			p = s->name;

		// exportname needs to see the name without the prefix too.
		if((fmtmode == FExp && !exportname(p)) || fmtmode == FDbg)
			return fmtprint(fp, "@\"%Z\".%s", s->pkg->path, p);

		return fmtstrcpy(fp, p);
	}

	return fmtstrcpy(fp, s->name);
}

static char*
basicnames[] =
{
	[TINT]		= "int",
	[TUINT]		= "uint",
	[TINT8]		= "int8",
	[TUINT8]	= "uint8",
	[TINT16]	= "int16",
	[TUINT16]	= "uint16",
	[TINT32]	= "int32",
	[TUINT32]	= "uint32",
	[TINT64]	= "int64",
	[TUINT64]	= "uint64",
	[TUINTPTR]	= "uintptr",
	[TFLOAT32]	= "float32",
	[TFLOAT64]	= "float64",
	[TCOMPLEX64]	= "complex64",
	[TCOMPLEX128]	= "complex128",
	[TBOOL]		= "bool",
	[TANY]		= "any",
	[TSTRING]	= "string",
	[TNIL]		= "nil",
	[TIDEAL]	= "untyped number",
	[TBLANK]	= "blank",
};

static int
typefmt(Fmt *fp, Type *t)
{
	Type *t1;
	Sym *s;

	if(t == T)
		return fmtstrcpy(fp, "<T>");

	if (t == bytetype || t == runetype) {
		// in %-T mode collapse rune and byte with their originals.
		if(fmtmode != FTypeId)
			return fmtprint(fp, "%hS", t->sym);
		t = types[t->etype];
	}

	if(t == errortype)
		return fmtstrcpy(fp, "error");

	// Unless the 'l' flag was specified, if the type has a name, just print that name.
	if(!(fp->flags&FmtLong) && t->sym && t->etype != TFIELD && t != types[t->etype]) {
		switch(fmtmode) {
		case FTypeId:
			if(fp->flags&FmtShort) {
				if(t->vargen)
					return fmtprint(fp, "%hS·%d", t->sym, t->vargen);
				return fmtprint(fp, "%hS", t->sym);
			}
			if(fp->flags&FmtUnsigned)
				return fmtprint(fp, "%uS", t->sym);
			// fallthrough
		case FExp:
			if(t->sym->pkg == localpkg && t->vargen)
				return fmtprint(fp, "%S·%d", t->sym, t->vargen);
			break;
		}
		return fmtprint(fp, "%S", t->sym);
	}

	if(t->etype < nelem(basicnames) && basicnames[t->etype] != nil) {
		if(fmtmode == FErr && (t == idealbool || t == idealstring))
			fmtstrcpy(fp, "untyped ");
		return fmtstrcpy(fp, basicnames[t->etype]);
	}

	if(fmtmode == FDbg)
		fmtprint(fp, "%E-", t->etype);

	switch(t->etype) {
	case TPTR32:
	case TPTR64:
		if(fmtmode == FTypeId && (fp->flags&FmtShort))
			return fmtprint(fp, "*%hT", t->type);
		return fmtprint(fp, "*%T", t->type);

	case TARRAY:
		if(t->bound >= 0)
			return fmtprint(fp, "[%lld]%T", t->bound, t->type);
		if(t->bound == -100)
			return fmtprint(fp, "[...]%T", t->type);
		return fmtprint(fp, "[]%T", t->type);

	case TCHAN:
		switch(t->chan) {
		case Crecv:
			return fmtprint(fp, "<-chan %T", t->type);
		case Csend:
			return fmtprint(fp, "chan<- %T", t->type);
		}

		if(t->type != T && t->type->etype == TCHAN && t->type->sym == S && t->type->chan == Crecv)
			return fmtprint(fp, "chan (%T)", t->type);
		return fmtprint(fp, "chan %T", t->type);

	case TMAP:
		return fmtprint(fp, "map[%T]%T", t->down, t->type);

	case TINTER:
		fmtstrcpy(fp, "interface {");
		for(t1=t->type; t1!=T; t1=t1->down)
			if(exportname(t1->sym->name)) {
				if(t1->down)
					fmtprint(fp, " %hS%hT;", t1->sym, t1->type);
				else
					fmtprint(fp, " %hS%hT ", t1->sym, t1->type);
			} else {
				// non-exported method names must be qualified
				if(t1->down)
					fmtprint(fp, " %uS%hT;", t1->sym, t1->type);
				else
					fmtprint(fp, " %uS%hT ", t1->sym, t1->type);
			}
		fmtstrcpy(fp, "}");
		return 0;

	case TFUNC:
		if(fp->flags & FmtShort) {
			fmtprint(fp, "%T", getinargx(t));
		} else {
			if(t->thistuple)
				fmtprint(fp, "method%T func%T", getthisx(t), getinargx(t));
			else
				fmtprint(fp, "func%T", getinargx(t));
		}
		switch(t->outtuple) {
		case 0:
			break;
		case 1:
			if(fmtmode != FExp) {
				fmtprint(fp, " %T", getoutargx(t)->type->type);	 // struct->field->field's type
				break;
			}
		default:
			fmtprint(fp, " %T", getoutargx(t));
			break;
		}
		return 0;

	case TSTRUCT:
		// Format the bucket struct for map[x]y as map.bucket[x]y.
		// This avoids a recursive print that generates very long names.
		if(t->map != T) {
			if(t->map->bucket == t) {
				return fmtprint(fp, "map.bucket[%T]%T", t->map->down, t->map->type);
			}
			if(t->map->hmap == t) {
				return fmtprint(fp, "map.hdr[%T]%T", t->map->down, t->map->type);
			}
			if(t->map->hiter == t) {
				return fmtprint(fp, "map.iter[%T]%T", t->map->down, t->map->type);
			}
			yyerror("unknown internal map type");
		}

		if(t->funarg) {
			fmtstrcpy(fp, "(");
			if(fmtmode == FTypeId || fmtmode == FErr) {	// no argument names on function signature, and no "noescape"/"nosplit" tags
				for(t1=t->type; t1!=T; t1=t1->down)
					if(t1->down)
						fmtprint(fp, "%hT, ", t1);
					else
						fmtprint(fp, "%hT", t1);
			} else {
				for(t1=t->type; t1!=T; t1=t1->down)
					if(t1->down)
						fmtprint(fp, "%T, ", t1);
					else
						fmtprint(fp, "%T", t1);
			}
			fmtstrcpy(fp, ")");
		} else {
			fmtstrcpy(fp, "struct {");
			for(t1=t->type; t1!=T; t1=t1->down)
				if(t1->down)
					fmtprint(fp, " %lT;", t1);
				else
					fmtprint(fp, " %lT ", t1);
			fmtstrcpy(fp, "}");
		}
		return 0;

	case TFIELD:
		if(!(fp->flags&FmtShort)) {
			s = t->sym;

			// Take the name from the original, lest we substituted it with ~r%d or ~b%d.
			// ~r%d is a (formerly) unnamed result.
			if ((fmtmode == FErr || fmtmode == FExp) && t->nname != N) {
				if(t->nname->orig != N) {
					s = t->nname->orig->sym;
					if(s != S && s->name[0] == '~') {
						if(s->name[1] == 'r') // originally an unnamed result
							s = S;
						else if(s->name[1] == 'b') // originally the blank identifier _
							s = lookup("_");
					}
				} else 
					s = S;
			}
			
			if(s != S && !t->embedded) {
				if(t->funarg)
					fmtprint(fp, "%N ", t->nname);
				else if(fp->flags&FmtLong)
					fmtprint(fp, "%hhS ", s);  // qualify non-exported names (used on structs, not on funarg)
				else 
					fmtprint(fp, "%S ", s);
			} else if(fmtmode == FExp) {
				// TODO(rsc) this breaks on the eliding of unused arguments in the backend
				// when this is fixed, the special case in dcl.c checkarglist can go.
				//if(t->funarg)
				//	fmtstrcpy(fp, "_ ");
				//else
				if(t->embedded && s->pkg != nil && s->pkg->path->len > 0)
					fmtprint(fp, "@\"%Z\".? ", s->pkg->path);
				else
					fmtstrcpy(fp, "? ");
			}
		}

		if(t->isddd)
			fmtprint(fp, "...%T", t->type->type);
		else
			fmtprint(fp, "%T", t->type);

		if(!(fp->flags&FmtShort) && t->note)
			fmtprint(fp, " \"%Z\"", t->note);
		return 0;

	case TFORW:
		if(t->sym)
			return fmtprint(fp, "undefined %S", t->sym);
		return fmtstrcpy(fp, "undefined");

	case TUNSAFEPTR:
		if(fmtmode == FExp)
			return fmtprint(fp, "@\"unsafe\".Pointer");
		return fmtprint(fp, "unsafe.Pointer");
	}

	if(fmtmode == FExp)
		fatal("missing %E case during export", t->etype);
	// Don't know how to handle - fall back to detailed prints.
	return fmtprint(fp, "%E <%S> %T", t->etype, t->sym, t->type);
}

// Statements which may be rendered with a simplestmt as init.
static int
stmtwithinit(int op)
{
	switch(op) {
	case OIF:
	case OFOR:
	case OSWITCH:
		return 1;
	}
	return 0;
}

static int
stmtfmt(Fmt *f, Node *n)
{
	int complexinit, simpleinit, extrablock;

	// some statements allow for an init, but at most one,
	// but we may have an arbitrary number added, eg by typecheck
	// and inlining.  If it doesn't fit the syntax, emit an enclosing
	// block starting with the init statements.

	// if we can just say "for" n->ninit; ... then do so
	simpleinit = n->ninit && !n->ninit->next && !n->ninit->n->ninit && stmtwithinit(n->op);
	// otherwise, print the inits as separate statements
	complexinit = n->ninit && !simpleinit && (fmtmode != FErr);
	// but if it was for if/for/switch, put in an extra surrounding block to limit the scope
	extrablock = complexinit && stmtwithinit(n->op);

	if(extrablock)
		fmtstrcpy(f, "{");

	if(complexinit)
		fmtprint(f, " %H; ", n->ninit);

	switch(n->op){
	case ODCL:
		if(fmtmode == FExp) {
			switch(n->left->class&~PHEAP) {
			case PPARAM:
			case PPARAMOUT:
			case PAUTO:
				fmtprint(f, "var %N %T", n->left, n->left->type);
				goto ret;
			}
		}			
		fmtprint(f, "var %S %T", n->left->sym, n->left->type);
		break;

	case ODCLFIELD:
		if(n->left)
			fmtprint(f, "%N %N", n->left, n->right);
		else
			fmtprint(f, "%N", n->right);
		break;

	case OAS:
		// Don't export "v = <N>" initializing statements, hope they're always 
		// preceded by the DCL which will be re-parsed and typecheck to reproduce
		// the "v = <N>" again.
		if(fmtmode == FExp && n->right == N)
			break;

		if(n->colas && !complexinit)
			fmtprint(f, "%N := %N", n->left, n->right);
		else
			fmtprint(f, "%N = %N", n->left, n->right);
		break;

	case OASOP:
		if(n->implicit) {
			if(n->etype == OADD)
				fmtprint(f, "%N++", n->left);
			else
				fmtprint(f, "%N--", n->left);
			break;
		}
		fmtprint(f, "%N %#O= %N", n->left, n->etype, n->right);
		break;

	case OAS2:
		if(n->colas && !complexinit) {
			fmtprint(f, "%,H := %,H", n->list, n->rlist);
			break;
		}
		// fallthrough
	case OAS2DOTTYPE:
	case OAS2FUNC:
	case OAS2MAPR:
	case OAS2RECV:
		fmtprint(f, "%,H = %,H", n->list, n->rlist);
		break;

	case ORETURN:
		fmtprint(f, "return %,H", n->list);
		break;

	case ORETJMP:
		fmtprint(f, "retjmp %S", n->sym);
		break;
	
	case OPROC:
		fmtprint(f, "go %N", n->left);
		break;

	case ODEFER:
		fmtprint(f, "defer %N", n->left);
		break;

	case OIF:
		if(simpleinit)
			fmtprint(f, "if %N; %N { %H }", n->ninit->n, n->ntest, n->nbody);
		else
			fmtprint(f, "if %N { %H }", n->ntest, n->nbody);
		if(n->nelse)
			fmtprint(f, " else { %H }", n->nelse);
		break;

	case OFOR:
		if(fmtmode == FErr) {	// TODO maybe only if FmtShort, same below
			fmtstrcpy(f, "for loop");
			break;
		}

		fmtstrcpy(f, "for");
		if(simpleinit)
			fmtprint(f, " %N;", n->ninit->n);
		else if(n->nincr)
			fmtstrcpy(f, " ;");

		if(n->ntest)
			fmtprint(f, " %N", n->ntest);

		if(n->nincr)
			fmtprint(f, "; %N", n->nincr);
		else if(simpleinit)
			fmtstrcpy(f, ";");


		fmtprint(f, " { %H }", n->nbody);
		break;

	case ORANGE:
		if(fmtmode == FErr) {
			fmtstrcpy(f, "for loop");
			break;
		}
		
		if(n->list == nil) {
			fmtprint(f, "for range %N { %H }", n->right, n->nbody);
			break;
		}
		fmtprint(f, "for %,H = range %N { %H }", n->list, n->right, n->nbody);
		break;

	case OSELECT:
	case OSWITCH:
		if(fmtmode == FErr) {
			fmtprint(f, "%O statement", n->op);
			break;
		}

		fmtprint(f, "%#O", n->op);
		if(simpleinit)
			fmtprint(f, " %N;", n->ninit->n);
		if(n->ntest)
			fmtprint(f, "%N", n->ntest);

		fmtprint(f, " { %H }", n->list);
		break;

	case OCASE:
	case OXCASE:
		if(n->list)
			fmtprint(f, "case %,H: %H", n->list, n->nbody);
		else
			fmtprint(f, "default: %H", n->nbody);
		break;

	case OBREAK:
	case OCONTINUE:
	case OGOTO:
	case OFALL:
	case OXFALL:
		if(n->left)
			fmtprint(f, "%#O %N", n->op, n->left);
		else
			fmtprint(f, "%#O", n->op);
		break;

	case OEMPTY:
		break;

	case OLABEL:
		fmtprint(f, "%N: ", n->left);
		break;
	  
	}
ret:

	if(extrablock)
		fmtstrcpy(f, "}");

	return 0;
}


static int opprec[] = {
	[OAPPEND] = 8,
	[OARRAYBYTESTR] = 8,
	[OARRAYLIT] = 8,
	[OARRAYRUNESTR] = 8,
	[OCALLFUNC] = 8,
	[OCALLINTER] = 8,
	[OCALLMETH] = 8,
	[OCALL] = 8,
	[OCAP] = 8,
	[OCLOSE] = 8,
	[OCONVIFACE] = 8,
	[OCONVNOP] = 8,
	[OCONV] = 8,
	[OCOPY] = 8,
	[ODELETE] = 8,
	[OLEN] = 8,
	[OLITERAL] = 8,
	[OMAKESLICE] = 8,
	[OMAKE] = 8,
	[OMAPLIT] = 8,
	[ONAME] = 8,
	[ONEW] = 8,
	[ONONAME] = 8,
	[OPACK] = 8,
	[OPANIC] = 8,
	[OPAREN] = 8,
	[OPRINTN] = 8,
	[OPRINT] = 8,
	[ORUNESTR] = 8,
	[OSTRARRAYBYTE] = 8,
	[OSTRARRAYRUNE] = 8,
	[OSTRUCTLIT] = 8,
	[OTARRAY] = 8,
	[OTCHAN] = 8,
	[OTFUNC] = 8,
	[OTINTER] = 8,
	[OTMAP] = 8,
	[OTSTRUCT] = 8,

	[OINDEXMAP] = 8,
	[OINDEX] = 8,
	[OSLICE] = 8,
	[OSLICESTR] = 8,
	[OSLICEARR] = 8,
	[OSLICE3] = 8,
	[OSLICE3ARR] = 8,
	[ODOTINTER] = 8,
	[ODOTMETH] = 8,
	[ODOTPTR] = 8,
	[ODOTTYPE2] = 8,
	[ODOTTYPE] = 8,
	[ODOT] = 8,
	[OXDOT] = 8,
	[OCALLPART] = 8,

	[OPLUS] = 7,
	[ONOT] = 7,
	[OCOM] = 7,
	[OMINUS] = 7,
	[OADDR] = 7,
	[OIND] = 7,
	[ORECV] = 7,

	[OMUL] = 6,
	[ODIV] = 6,
	[OMOD] = 6,
	[OLSH] = 6,
	[ORSH] = 6,
	[OAND] = 6,
	[OANDNOT] = 6,

	[OADD] = 5,
	[OSUB] = 5,
	[OOR] = 5,
	[OXOR] = 5,

	[OEQ] = 4,
	[OLT] = 4,
	[OLE] = 4,
	[OGE] = 4,
	[OGT] = 4,
	[ONE] = 4,
	[OCMPSTR] = 4,
	[OCMPIFACE] = 4,

	[OSEND] = 3,
	[OANDAND] = 2,
	[OOROR] = 1,

	// Statements handled by stmtfmt
	[OAS] = -1,
	[OAS2] = -1,
	[OAS2DOTTYPE] = -1,
	[OAS2FUNC] = -1,
	[OAS2MAPR] = -1,
	[OAS2RECV] = -1,
	[OASOP] = -1,
	[OBREAK] = -1,
	[OCASE] = -1,
	[OCONTINUE] = -1,
	[ODCL] = -1,
	[ODCLFIELD] = -1,
	[ODEFER] = -1,
	[OEMPTY] = -1,
	[OFALL] = -1,
	[OFOR] = -1,
	[OGOTO] = -1,
	[OIF] = -1,
	[OLABEL] = -1,
	[OPROC] = -1,
	[ORANGE] = -1,
	[ORETURN] = -1,
	[OSELECT] = -1,
	[OSWITCH] = -1,
	[OXCASE] = -1,
	[OXFALL] = -1,

	[OEND] = 0
};

static int
exprfmt(Fmt *f, Node *n, int prec)
{
	int nprec;
	int ptrlit;
	NodeList *l;

	while(n && n->implicit && (n->op == OIND || n->op == OADDR))
		n = n->left;

	if(n == N)
		return fmtstrcpy(f, "<N>");

	nprec = opprec[n->op];
	if(n->op == OTYPE && n->sym != S)
		nprec = 8;

	if(prec > nprec)
		return fmtprint(f, "(%N)", n);

	switch(n->op) {
	case OPAREN:
		return fmtprint(f, "(%N)", n->left);

	case ODDDARG:
		return fmtprint(f, "... argument");

	case OREGISTER:
		return fmtprint(f, "%R", n->val.u.reg);

	case OLITERAL:  // this is a bit of a mess
		if(fmtmode == FErr && n->sym != S)
			return fmtprint(f, "%S", n->sym);
		if(n->val.ctype == CTNIL && n->orig != N && n->orig != n)
			return exprfmt(f, n->orig, prec);
		if(n->type != T && n->type != types[n->type->etype] && n->type != idealbool && n->type != idealstring) {
			// Need parens when type begins with what might
			// be misinterpreted as a unary operator: * or <-.
			if(isptr[n->type->etype] || (n->type->etype == TCHAN && n->type->chan == Crecv))
				return fmtprint(f, "(%T)(%V)", n->type, &n->val);
			else 
				return fmtprint(f, "%T(%V)", n->type, &n->val);
		}
		return fmtprint(f, "%V", &n->val);

	case ONAME:
		// Special case: name used as local variable in export.
		// _ becomes ~b%d internally; print as _ for export
		if(fmtmode == FExp && n->sym && n->sym->name[0] == '~' && n->sym->name[1] == 'b')
			return fmtprint(f, "_");
		if(fmtmode == FExp && n->sym && !isblank(n) && n->vargen > 0)
			return fmtprint(f, "%S·%d", n->sym, n->vargen);

		// Special case: explicit name of func (*T) method(...) is turned into pkg.(*T).method,
		// but for export, this should be rendered as (*pkg.T).meth.
		// These nodes have the special property that they are names with a left OTYPE and a right ONAME.
		if(fmtmode == FExp && n->left && n->left->op == OTYPE && n->right && n->right->op == ONAME) {
			if(isptr[n->left->type->etype])
				return fmtprint(f, "(%T).%hhS", n->left->type, n->right->sym);
			else
				return fmtprint(f, "%T.%hhS", n->left->type, n->right->sym);
		}
		//fallthrough
	case OPACK:
	case ONONAME:
		return fmtprint(f, "%S", n->sym);

	case OTYPE:
		if(n->type == T && n->sym != S)
			return fmtprint(f, "%S", n->sym);
		return fmtprint(f, "%T", n->type);

	case OTARRAY:
		if(n->left)
			return fmtprint(f, "[]%N", n->left);
		return fmtprint(f, "[]%N", n->right);  // happens before typecheck

	case OTMAP:
		return fmtprint(f, "map[%N]%N", n->left, n->right);

	case OTCHAN:
		switch(n->etype) {
		case Crecv:
			return fmtprint(f, "<-chan %N", n->left);
		case Csend:
			return fmtprint(f, "chan<- %N", n->left);
		default:
			if(n->left != N && n->left->op == OTCHAN && n->left->sym == S && n->left->etype == Crecv)
				return fmtprint(f, "chan (%N)", n->left);
			else
				return fmtprint(f, "chan %N", n->left);
		}

	case OTSTRUCT:
		return fmtprint(f, "<struct>");

	case OTINTER:
		return fmtprint(f, "<inter>");

	case OTFUNC:
		return fmtprint(f, "<func>");

	case OCLOSURE:
		if(fmtmode == FErr)
			return fmtstrcpy(f, "func literal");
		if(n->nbody)
			return fmtprint(f, "%T { %H }", n->type, n->nbody);
		return fmtprint(f, "%T { %H }", n->type, n->closure->nbody);

	case OCOMPLIT:
		ptrlit = n->right != N && n->right->implicit && n->right->type && isptr[n->right->type->etype];
		if(fmtmode == FErr) {
			if(n->right != N && n->right->type != T && !n->implicit) {
				if(ptrlit)
					return fmtprint(f, "&%T literal", n->right->type->type);
				else
					return fmtprint(f, "%T literal", n->right->type);
			}
			return fmtstrcpy(f, "composite literal");
		}
		if(fmtmode == FExp && ptrlit)
			// typecheck has overwritten OIND by OTYPE with pointer type.
			return fmtprint(f, "(&%T{ %,H })", n->right->type->type, n->list);
		return fmtprint(f, "(%N{ %,H })", n->right, n->list);

	case OPTRLIT:
		if(fmtmode == FExp && n->left->implicit)
			return fmtprint(f, "%N", n->left);
		return fmtprint(f, "&%N", n->left);

	case OSTRUCTLIT:
		if(fmtmode == FExp) {   // requires special handling of field names
			if(n->implicit)
				fmtstrcpy(f, "{");
			else
				fmtprint(f, "(%T{", n->type);
			for(l=n->list; l; l=l->next) {
				fmtprint(f, " %hhS:%N", l->n->left->sym, l->n->right);

				if(l->next)
					fmtstrcpy(f, ",");
				else
					fmtstrcpy(f, " ");
			}
			if(!n->implicit)
				return fmtstrcpy(f, "})");
			return fmtstrcpy(f, "}");
		}
		// fallthrough

	case OARRAYLIT:
	case OMAPLIT:
		if(fmtmode == FErr)
			return fmtprint(f, "%T literal", n->type);
		if(fmtmode == FExp && n->implicit)
			return fmtprint(f, "{ %,H }", n->list);
		return fmtprint(f, "(%T{ %,H })", n->type, n->list);

	case OKEY:
		if(n->left && n->right) {
			if(fmtmode == FExp && n->left->type && n->left->type->etype == TFIELD) {
				// requires special handling of field names
				return fmtprint(f, "%hhS:%N", n->left->sym, n->right);
			} else
				return fmtprint(f, "%N:%N", n->left, n->right);
		}
		if(!n->left && n->right)
			return fmtprint(f, ":%N", n->right);
		if(n->left && !n->right)
			return fmtprint(f, "%N:", n->left);
		return fmtstrcpy(f, ":");

	case OXDOT:
	case ODOT:
	case ODOTPTR:
	case ODOTINTER:
	case ODOTMETH:
	case OCALLPART:
		exprfmt(f, n->left, nprec);
		if(n->right == N || n->right->sym == S)
			return fmtstrcpy(f, ".<nil>");
		return fmtprint(f, ".%hhS", n->right->sym);

	case ODOTTYPE:
	case ODOTTYPE2:
		exprfmt(f, n->left, nprec);
		if(n->right != N)
			return fmtprint(f, ".(%N)", n->right);
		return fmtprint(f, ".(%T)", n->type);

	case OINDEX:
	case OINDEXMAP:
	case OSLICE:
	case OSLICESTR:
	case OSLICEARR:
	case OSLICE3:
	case OSLICE3ARR:
		exprfmt(f, n->left, nprec);
		return fmtprint(f, "[%N]", n->right);

	case OCOPY:
	case OCOMPLEX:
		return fmtprint(f, "%#O(%N, %N)", n->op, n->left, n->right);

	case OCONV:
	case OCONVIFACE:
	case OCONVNOP:
	case OARRAYBYTESTR:
	case OARRAYRUNESTR:
	case OSTRARRAYBYTE:
	case OSTRARRAYRUNE:
	case ORUNESTR:
		if(n->type == T || n->type->sym == S)
			return fmtprint(f, "(%T)(%N)", n->type, n->left);
		if(n->left)
			return fmtprint(f, "%T(%N)", n->type, n->left);
		return fmtprint(f, "%T(%,H)", n->type, n->list);

	case OREAL:
	case OIMAG:
	case OAPPEND:
	case OCAP:
	case OCLOSE:
	case ODELETE:
	case OLEN:
	case OMAKE:
	case ONEW:
	case OPANIC:
	case ORECOVER:
	case OPRINT:
	case OPRINTN:
		if(n->left)
			return fmtprint(f, "%#O(%N)", n->op, n->left);
		if(n->isddd)
			return fmtprint(f, "%#O(%,H...)", n->op, n->list);
		return fmtprint(f, "%#O(%,H)", n->op, n->list);

	case OCALL:
	case OCALLFUNC:
	case OCALLINTER:
	case OCALLMETH:
		exprfmt(f, n->left, nprec);
		if(n->isddd)
			return fmtprint(f, "(%,H...)", n->list);
		return fmtprint(f, "(%,H)", n->list);

	case OMAKEMAP:
	case OMAKECHAN:
	case OMAKESLICE:
		if(n->list) // pre-typecheck
			return fmtprint(f, "make(%T, %,H)", n->type, n->list);
		if(n->right)
			return fmtprint(f, "make(%T, %N, %N)", n->type, n->left, n->right);
		if(n->left)
			return fmtprint(f, "make(%T, %N)", n->type, n->left);
		return fmtprint(f, "make(%T)", n->type);

	// Unary
	case OPLUS:
	case OMINUS:
	case OADDR:
	case OCOM:
	case OIND:
	case ONOT:
	case ORECV:
		if(n->left->op == n->op)
			fmtprint(f, "%#O ", n->op);
		else
			fmtprint(f, "%#O", n->op);
		return exprfmt(f, n->left, nprec+1);

	// Binary
	case OADD:
	case OAND:
	case OANDAND:
	case OANDNOT:
	case ODIV:
	case OEQ:
	case OGE:
	case OGT:
	case OLE:
	case OLT:
	case OLSH:
	case OMOD:
	case OMUL:
	case ONE:
	case OOR:
	case OOROR:
	case ORSH:
	case OSEND:
	case OSUB:
	case OXOR:
		exprfmt(f, n->left, nprec);
		fmtprint(f, " %#O ", n->op);
		exprfmt(f, n->right, nprec+1);
		return 0;

	case OADDSTR:
		for(l=n->list; l; l=l->next) {
			if(l != n->list)
				fmtprint(f, " + ");
			exprfmt(f, l->n, nprec);
		}
		return 0;

	case OCMPSTR:
	case OCMPIFACE:
		exprfmt(f, n->left, nprec);
		fmtprint(f, " %#O ", n->etype);
		exprfmt(f, n->right, nprec+1);
		return 0;
	}

	return fmtprint(f, "<node %O>", n->op);
}

static int
nodefmt(Fmt *f, Node *n)
{
	Type *t;

	t = n->type;

	// we almost always want the original, except in export mode for literals
	// this saves the importer some work, and avoids us having to redo some
	// special casing for package unsafe
	if((fmtmode != FExp || n->op != OLITERAL) && n->orig != N)
		n = n->orig;

	if(f->flags&FmtLong && t != T) {
		if(t->etype == TNIL)
			return fmtprint(f, "nil");
		else
			return fmtprint(f, "%N (type %T)", n, t);
	}

	// TODO inlining produces expressions with ninits. we can't print these yet.

	if(opprec[n->op] < 0)
		return stmtfmt(f, n);

	return exprfmt(f, n, 0);
}

static int dumpdepth;

static void
indent(Fmt *fp)
{
	int i;

	fmtstrcpy(fp, "\n");
	for(i = 0; i < dumpdepth; ++i)
		fmtstrcpy(fp, ".   ");
}

static int
nodedump(Fmt *fp, Node *n)
{
	int recur;

	if(n == N)
		return 0;

	recur = !(fp->flags&FmtShort);

	if(recur) {
		indent(fp);
		if(dumpdepth > 10)
			return fmtstrcpy(fp, "...");

		if(n->ninit != nil) {
			fmtprint(fp, "%O-init%H", n->op, n->ninit);
			indent(fp);
		}
	}

//	fmtprint(fp, "[%p]", n);

	switch(n->op) {
	default:
		fmtprint(fp, "%O%J", n->op, n);
		break;
	case OREGISTER:
	case OINDREG:
		fmtprint(fp, "%O-%R%J", n->op, n->val.u.reg, n);
		break;
	case OLITERAL:
		fmtprint(fp, "%O-%V%J", n->op, &n->val, n);
		break;
	case ONAME:
	case ONONAME:
		if(n->sym != S)
			fmtprint(fp, "%O-%S%J", n->op, n->sym, n);
		else
			fmtprint(fp, "%O%J", n->op, n);
		if(recur && n->type == T && n->ntype) {
			indent(fp);
			fmtprint(fp, "%O-ntype%N", n->op, n->ntype);
		}
		break;
	case OASOP:
		fmtprint(fp, "%O-%O%J", n->op, n->etype, n);
		break;
	case OTYPE:
		fmtprint(fp, "%O %S%J type=%T", n->op, n->sym, n, n->type);
		if(recur && n->type == T && n->ntype) {
			indent(fp);
			fmtprint(fp, "%O-ntype%N", n->op, n->ntype);
		}
		break;
	}

	if(n->sym != S && n->op != ONAME)
		fmtprint(fp, " %S G%d", n->sym, n->vargen);

	if(n->type != T)
		fmtprint(fp, " %T", n->type);

	if(recur) {
		if(n->left)
			fmtprint(fp, "%N", n->left);
		if(n->right)
			fmtprint(fp, "%N", n->right);
		if(n->list) {
			indent(fp);
			fmtprint(fp, "%O-list%H", n->op, n->list);
		}
		if(n->rlist) {
			indent(fp);
			fmtprint(fp, "%O-rlist%H", n->op, n->rlist);
		}
		if(n->ntest) {
			indent(fp);
			fmtprint(fp, "%O-test%N", n->op, n->ntest);
		}
		if(n->nbody) {
			indent(fp);
			fmtprint(fp, "%O-body%H", n->op, n->nbody);
		}
		if(n->nelse) {
			indent(fp);
			fmtprint(fp, "%O-else%H", n->op, n->nelse);
		}
		if(n->nincr) {
			indent(fp);
			fmtprint(fp, "%O-incr%N", n->op, n->nincr);
		}
	}

	return 0;
}

// Fmt "%S": syms
// Flags:  "%hS" suppresses qualifying with package
static int
Sconv(Fmt *fp)
{
	Sym *s;
	int r, sm;
	unsigned long sf;

	if(fp->flags&FmtLong)
		return linksymfmt(fp);

	s = va_arg(fp->args, Sym*);
	if(s == S)
		return fmtstrcpy(fp, "<S>");

	if(s->name && s->name[0] == '_' && s->name[1] == '\0')
		return fmtstrcpy(fp, "_");

	sf = fp->flags;
	sm = setfmode(&fp->flags);
	r = symfmt(fp, s);
	fp->flags = sf;
	fmtmode = sm;
	return r;
}

// Fmt "%T": types.
// Flags: 'l' print definition, not name
//	  'h' omit 'func' and receiver from function types, short type names
//	  'u' package name, not prefix (FTypeId mode, sticky)
static int
Tconv(Fmt *fp)
{
	Type *t;
	int r, sm;
	unsigned long sf;

	t = va_arg(fp->args, Type*);
	if(t == T)
		return fmtstrcpy(fp, "<T>");

	if(t->trecur > 4)
		return fmtstrcpy(fp, "<...>");

	t->trecur++;
	sf = fp->flags;
	sm = setfmode(&fp->flags);

	if(fmtmode == FTypeId && (sf&FmtUnsigned))
		fmtpkgpfx++;
	if(fmtpkgpfx)
		fp->flags |= FmtUnsigned;

	r = typefmt(fp, t);

	if(fmtmode == FTypeId && (sf&FmtUnsigned))
		fmtpkgpfx--;

	fp->flags = sf;
	fmtmode = sm;
	t->trecur--;
	return r;
}

// Fmt '%N': Nodes.
// Flags: 'l' suffix with "(type %T)" where possible
//	  '+h' in debug mode, don't recurse, no multiline output
static int
Nconv(Fmt *fp)
{
	Node *n;
	int r, sm;
	unsigned long sf;

	n = va_arg(fp->args, Node*);
	if(n == N)
		return fmtstrcpy(fp, "<N>");
	sf = fp->flags;
	sm = setfmode(&fp->flags);

	r = -1;
	switch(fmtmode) {
	case FErr:
	case FExp:
		r = nodefmt(fp, n);
		break;
	case FDbg:
		dumpdepth++;
		r = nodedump(fp, n);
		dumpdepth--;
		break;
	default:
		fatal("unhandled %%N mode");
	}

	fp->flags = sf;
	fmtmode = sm;
	return r;
}

// Fmt '%H': NodeList.
// Flags: all those of %N plus ',': separate with comma's instead of semicolons.
static int
Hconv(Fmt *fp)
{
	NodeList *l;
	int r, sm;
	unsigned long sf;
	char *sep;

	l = va_arg(fp->args, NodeList*);

	if(l == nil && fmtmode == FDbg)
		return fmtstrcpy(fp, "<nil>");

	sf = fp->flags;
	sm = setfmode(&fp->flags);
	r = 0;
	sep = "; ";
	if(fmtmode == FDbg)
		sep = "\n";
	else if(fp->flags & FmtComma)
		sep = ", ";

	for(;l; l=l->next) {
		r += fmtprint(fp, "%N", l->n);
		if(l->next)
			r += fmtstrcpy(fp, sep);
	}

	fp->flags = sf;
	fmtmode = sm;
	return r;
}

void
fmtinstallgo(void)
{
	fmtmode = FErr;
	fmtinstall('E', Econv);		// etype opcodes
	fmtinstall('J', Jconv);		// all the node flags
	fmtinstall('H', Hconv);		// node lists
	fmtinstall('L', Lconv);		// line number
	fmtinstall('N', Nconv);		// node pointer
	fmtinstall('O', Oconv);		// node opcodes
	fmtinstall('S', Sconv);		// sym pointer
	fmtinstall('T', Tconv);		// type pointer
	fmtinstall('V', Vconv);		// val pointer
	fmtinstall('Z', Zconv);		// escaped string

	// These are in mparith1.c
	fmtinstall('B', Bconv);	// big numbers
	fmtinstall('F', Fconv);	// big float numbers

}

void
dumplist(char *s, NodeList *l)
{
	print("%s%+H\n", s, l);
}

void
dump(char *s, Node *n)
{
	print("%s [%p]%+N\n", s, n, n);
}

//   cmd/cc/godefs.cc
//
//   derived from pickle.cc which itself was derived from acid.cc.
//
//	Copyright © 1994-1999 Lucent Technologies Inc. All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009-2011 The Go Authors.	All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <u.h>
#include "cc.h"

static int upper;

static char *kwd[] =
{
	"_bool",
	"_break",
	"_byte",
	"_case",
	"_chan",
	"_complex128",
	"_complex64",
	"_const",
	"_continue",
	"_default",
	"_defer",
	"_else",
	"_fallthrough",
	"_false",
	"_float32",
	"_float64",
	"_for",
	"_func",
	"_go",
	"_goto",
	"_if",
	"_import",
	"_int",
	"_int16",
	"_int32",
	"_int64",
	"_int8",
	"_interface",
	"_intptr",
	"_map",
	"_package",
	"_panic",
	"_range",
	"_return",
	"_select",
	"_string",
	"_struct",
	"_switch",
	"_true",
	"_type",
	"_uint",
	"_uint16",
	"_uint32",
	"_uint64",
	"_uint8",
	"_uintptr",
	"_var",
};

static char*
pmap(char *s)
{
	int i, bot, top, mid;

	bot = -1;
	top = nelem(kwd);
	while(top - bot > 1){
		mid = (bot + top) / 2;
		i = strcmp(kwd[mid]+1, s);
		if(i == 0)
			return kwd[mid];
		if(i < 0)
			bot = mid;
		else
			top = mid;
	}

	return s;
}


int
Uconv(Fmt *fp)
{
	char str[STRINGSZ+1];
	char *s, *n;
	int i;

	str[0] = 0;
	s = va_arg(fp->args, char*);

	// strip package name
	n = strrchr(s, '.');
	if(n != nil)
		s = n + 1;

	if(s && *s) {
		if(upper)
			str[0] = toupper((uchar)*s);
		else
			str[0] = tolower((uchar)*s);
		for(i = 1; i < STRINGSZ && s[i] != 0; i++)
			str[i] = tolower((uchar)s[i]);
		str[i] = 0;
	}

	return fmtstrcpy(fp, pmap(str));
}


static Sym*
findsue(Type *t)
{
	int h;
	Sym *s;

	if(t != T)
	for(h=0; h<nelem(hash); h++)
		for(s = hash[h]; s != S; s = s->link)
			if(s->suetag && s->suetag->link == t)
				return s;
	return 0;
}

static void
printtypename(Type *t)
{
	Sym *s;
	int w;
	char *n;

	for( ; t != nil; t = t->link) {
		switch(t->etype) {
		case TIND:
			// Special handling of *void.
			if(t->link != nil && t->link->etype==TVOID) {
				Bprint(&outbuf, "unsafe.Pointer");
				return;
			}
			// *func == func
			if(t->link != nil && t->link->etype==TFUNC)
				continue;
			Bprint(&outbuf, "*");
			continue;
		case TARRAY:
			w = t->width;
			if(t->link && t->link->width)
				w /= t->link->width;
			Bprint(&outbuf, "[%d]", w);
			continue;
		}
		break;
	}

	if(t == nil) {
		Bprint(&outbuf, "bad // should not happen");
		return;
	}

	switch(t->etype) {
	case TINT:
	case TUINT:
	case TCHAR:
	case TUCHAR:
	case TSHORT:
	case TUSHORT:
	case TLONG:
	case TULONG:
	case TVLONG:
	case TUVLONG:
	case TFLOAT:
	case TDOUBLE:
		// All names used in the runtime code should be typedefs.
		if(t->tag != nil) {
			if(strcmp(t->tag->name, "intgo") == 0)
				Bprint(&outbuf, "int");
			else if(strcmp(t->tag->name, "uintgo") == 0)
				Bprint(&outbuf, "uint");
			else
				Bprint(&outbuf, "%s", t->tag->name);
		} else	
			Bprint(&outbuf, "C.%T", t);
		break;
	case TUNION:
	case TSTRUCT:
		s = findsue(t->link);
		n = "bad";
		if(s != S)
			n = s->name;
		else if(t->tag)
			n = t->tag->name;
		if(strcmp(n, "String") == 0)
			Bprint(&outbuf, "string");
		else if(strcmp(n, "Slice") == 0)
			Bprint(&outbuf, "[]byte");
		else if(strcmp(n, "Eface") == 0)
			Bprint(&outbuf, "interface{}");
		else
			Bprint(&outbuf, "%U", n);
		break;
	case TFUNC:
		// There's no equivalent to a C function in the Go world.
		Bprint(&outbuf, "unsafe.Pointer");
		break;
	case TDOT:
		Bprint(&outbuf, "...interface{}");
		break;
	default:
		Bprint(&outbuf, " weird<%T>", t);
	}
}

static int
dontrun(void)
{
	Io *i;
	int n;

	if(!debug['q'] && !debug['Q'])
		return 1;
	if(debug['q'] + debug['Q'] > 1) {
		n = 0;
		for(i=iostack; i; i=i->link)
			n++;
		if(n > 1)
			return 1;
	}

	upper = debug['Q'];
	return 0;
}

void
godeftype(Type *t)
{
	Sym *s;
	Type *l;
	int gotone;

	if(dontrun())
		return;

	switch(t->etype) {
	case TUNION:
	case TSTRUCT:
		s = findsue(t->link);
		if(s == S) {
			Bprint(&outbuf, "/* can't find %T */\n\n", t);
			return;
		}

		gotone = 0; // for unions, take first member of size equal to union
		Bprint(&outbuf, "type %U struct {\n", s->name);
		for(l = t->link; l != T; l = l->down) {
			Bprint(&outbuf, "\t");
			if(t->etype == TUNION) {
				if(!gotone && l->width == t->width)
					gotone = 1;
				else
					Bprint(&outbuf, "// (union)\t");
			}
			if(l->sym != nil)  // not anonymous field
				Bprint(&outbuf, "%U\t", l->sym->name);
			printtypename(l);
			Bprint(&outbuf, "\n");
		}
		Bprint(&outbuf, "}\n\n");
		break;

	default:
		Bprint(&outbuf, "/* %T */\n\n", t);
		break;
	}
}

void
godefvar(Sym *s)
{
	Type *t, *t1;
	char n;

	if(dontrun())
		return;

	t = s->type;
	if(t == nil)
		return;

	switch(t->etype) {
	case TENUM:
		if(!typefd[t->etype])
			Bprint(&outbuf, "const %s = %lld\n", s->name, s->vconst);
		else
			Bprint(&outbuf, "const %s = %f\n;", s->name, s->fconst);
		break;

	case TFUNC:
		Bprint(&outbuf, "func %U(", s->name);
		n = 'a';
		for(t1 = t->down; t1 != T; t1 = t1->down) {
			if(t1->etype == TVOID)
				break;
			if(t1 != t->down)
				Bprint(&outbuf, ", ");
			Bprint(&outbuf, "%c ", n++);
			printtypename(t1);
		}
		Bprint(&outbuf, ")");
		if(t->link && t->link->etype != TVOID) {
			Bprint(&outbuf, " ");
			printtypename(t->link);
		}
		Bprint(&outbuf, "\n");
		break;

	default:
		switch(s->class) {
		case CTYPEDEF:
			if(!typesu[t->etype]) {
				Bprint(&outbuf, "// type %U\t", s->name);
				printtypename(t);
				Bprint(&outbuf, "\n");
			}
			break;
		case CSTATIC:
		case CEXTERN:
		case CGLOBL:
			if(strchr(s->name, '$') != nil)
				break;
			if(strncmp(s->name, "go.weak.", 8) == 0)
				break;
			Bprint(&outbuf, "var %U\t", s->name);
			printtypename(t);
			Bprint(&outbuf, "\n");
			break;
		}
		break;
	}
}

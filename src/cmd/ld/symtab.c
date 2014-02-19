// Inferno utils/6l/span.c
// http://code.google.com/p/inferno-os/source/browse/utils/6l/span.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
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

// Symbol table.

#include	"l.h"
#include	"../ld/lib.h"
#include	"../ld/elf.h"

static int maxelfstr;

static int
putelfstr(char *s)
{
	int off, n;

	if(elfstrsize == 0 && s[0] != 0) {
		// first entry must be empty string
		putelfstr("");
	}

	n = strlen(s)+1;
	if(elfstrsize+n > maxelfstr) {
		maxelfstr = 2*(elfstrsize+n+(1<<20));
		elfstrdat = realloc(elfstrdat, maxelfstr);
	}
	off = elfstrsize;
	elfstrsize += n;
	memmove(elfstrdat+off, s, n);
	return off;
}

static void
putelfsyment(int off, vlong addr, vlong size, int info, int shndx, int other)
{
	switch(thechar) {
	case '6':
		LPUT(off);
		cput(info);
		cput(other);
		WPUT(shndx);
		VPUT(addr);
		VPUT(size);
		symsize += ELF64SYMSIZE;
		break;
	default:
		LPUT(off);
		LPUT(addr);
		LPUT(size);
		cput(info);
		cput(other);
		WPUT(shndx);
		symsize += ELF32SYMSIZE;
		break;
	}
}

static int numelfsym = 1; // 0 is reserved
static int elfbind;

static void
putelfsym(LSym *x, char *s, int t, vlong addr, vlong size, int ver, LSym *go)
{
	int bind, type, off;
	LSym *xo;

	USED(go);
	switch(t) {
	default:
		return;
	case 'T':
		type = STT_FUNC;
		break;
	case 'D':
		type = STT_OBJECT;
		break;
	case 'B':
		type = STT_OBJECT;
		break;
	}
	xo = x;
	while(xo->outer != nil)
		xo = xo->outer;
	if(xo->sect == nil) {
		ctxt->cursym = x;
		diag("missing section in putelfsym");
		return;
	}
	if(xo->sect->elfsect == nil) {
		ctxt->cursym = x;
		diag("missing ELF section in putelfsym");
		return;
	}

	// One pass for each binding: STB_LOCAL, STB_GLOBAL,
	// maybe one day STB_WEAK.
	bind = STB_GLOBAL;
	if(ver || (x->type & SHIDDEN))
		bind = STB_LOCAL;

	// In external linking mode, we have to invoke gcc with -rdynamic
	// to get the exported symbols put into the dynamic symbol table.
	// To avoid filling the dynamic table with lots of unnecessary symbols,
	// mark all Go symbols local (not global) in the final executable.
	if(linkmode == LinkExternal && !(x->cgoexport&CgoExportStatic))
		bind = STB_LOCAL;

	if(bind != elfbind)
		return;

	off = putelfstr(s);
	if(linkmode == LinkExternal)
		addr -= xo->sect->vaddr;
	putelfsyment(off, addr, size, (bind<<4)|(type&0xf), xo->sect->elfsect->shnum, (x->type & SHIDDEN) ? 2 : 0);
	x->elfsym = numelfsym++;
}

void
putelfsectionsym(LSym* s, int shndx)
{
	putelfsyment(0, 0, 0, (STB_LOCAL<<4)|STT_SECTION, shndx, 0);
	s->elfsym = numelfsym++;
}

void
putelfsymshndx(vlong sympos, int shndx)
{
	vlong here;

	here = cpos();
	switch(thechar) {
	case '6':
		cseek(sympos+6);
		break;
	default:
		cseek(sympos+14);
		break;
	}
	WPUT(shndx);
	cseek(here);
}

void
asmelfsym(void)
{
	LSym *s;
	char *name;

	// the first symbol entry is reserved
	putelfsyment(0, 0, 0, (STB_LOCAL<<4)|STT_NOTYPE, 0, 0);

	dwarfaddelfsectionsyms();

	elfbind = STB_LOCAL;
	genasmsym(putelfsym);
	
	if(linkmode == LinkExternal && HEADTYPE != Hopenbsd) {
		s = linklookup(ctxt, "runtime.tlsgm", 0);
		if(s->sect == nil) {
			ctxt->cursym = nil;
			diag("missing section for %s", s->name);
			errorexit();
		}
		putelfsyment(putelfstr(s->name), 0, 2*PtrSize, (STB_LOCAL<<4)|STT_TLS, s->sect->elfsect->shnum, 0);
		s->elfsym = numelfsym++;
	}

	elfbind = STB_GLOBAL;
	elfglobalsymndx = numelfsym;
	genasmsym(putelfsym);
	
	for(s=ctxt->allsym; s!=S; s=s->allsym) {
		if(s->type != SHOSTOBJ && s->type != SDYNIMPORT)
			continue;
		if(s->type == SDYNIMPORT)
			name = s->extname;
		else
			name = s->name;
		putelfsyment(putelfstr(name), 0, 0, (STB_GLOBAL<<4)|STT_NOTYPE, 0, 0);
		s->elfsym = numelfsym++;
	}
}

static void
putplan9sym(LSym *x, char *s, int t, vlong addr, vlong size, int ver, LSym *go)
{
	int i, l;

	USED(go);
	USED(ver);
	USED(size);
	USED(x);
	switch(t) {
	case 'T':
	case 'L':
	case 'D':
	case 'B':
		if(ver)
			t += 'a' - 'A';
	case 'a':
	case 'p':
	case 'f':
	case 'z':
	case 'Z':
	case 'm':
		l = 4;
		if(HEADTYPE == Hplan9 && thechar == '6' && !debug['8']) {
			lputb(addr>>32);
			l = 8;
		}
		lputb(addr);
		cput(t+0x80); /* 0x80 is variable length */

		if(t == 'z' || t == 'Z') {
			cput(s[0]);
			for(i=1; s[i] != 0 || s[i+1] != 0; i += 2) {
				cput(s[i]);
				cput(s[i+1]);
			}
			cput(0);
			cput(0);
			i++;
		} else {
			/* skip the '<' in filenames */
			if(t == 'f')
				s++;
			for(i=0; s[i]; i++)
				cput(s[i]);
			cput(0);
		}
		symsize += l + 1 + i + 1;
		break;
	default:
		return;
	};
}

void
asmplan9sym(void)
{
	genasmsym(putplan9sym);
}

static LSym *symt;

void
wputl(ushort w)
{
	cput(w);
	cput(w>>8);
}

void
wputb(ushort w)
{
	cput(w>>8);
	cput(w);
}

void
lputb(int32 l)
{
	cput(l>>24);
	cput(l>>16);
	cput(l>>8);
	cput(l);
}

void
lputl(int32 l)
{
	cput(l);
	cput(l>>8);
	cput(l>>16);
	cput(l>>24);
}

void
vputb(uint64 v)
{
	lputb(v>>32);
	lputb(v);
}

void
vputl(uint64 v)
{
	lputl(v);
	lputl(v >> 32);
}

void
symtab(void)
{
	LSym *s, *symtype, *symtypelink, *symgostring, *symgofunc;

	dosymtype();

	// Define these so that they'll get put into the symbol table.
	// data.c:/^address will provide the actual values.
	xdefine("text", STEXT, 0);
	xdefine("etext", STEXT, 0);
	xdefine("typelink", SRODATA, 0);
	xdefine("etypelink", SRODATA, 0);
	xdefine("rodata", SRODATA, 0);
	xdefine("erodata", SRODATA, 0);
	xdefine("noptrdata", SNOPTRDATA, 0);
	xdefine("enoptrdata", SNOPTRDATA, 0);
	xdefine("data", SDATA, 0);
	xdefine("edata", SDATA, 0);
	xdefine("bss", SBSS, 0);
	xdefine("ebss", SBSS, 0);
	xdefine("noptrbss", SNOPTRBSS, 0);
	xdefine("enoptrbss", SNOPTRBSS, 0);
	xdefine("end", SBSS, 0);
	xdefine("epclntab", SRODATA, 0);
	xdefine("esymtab", SRODATA, 0);

	// garbage collection symbols
	s = linklookup(ctxt, "gcdata", 0);
	s->type = SRODATA;
	s->size = 0;
	s->reachable = 1;
	xdefine("egcdata", SRODATA, 0);

	s = linklookup(ctxt, "gcbss", 0);
	s->type = SRODATA;
	s->size = 0;
	s->reachable = 1;
	xdefine("egcbss", SRODATA, 0);

	// pseudo-symbols to mark locations of type, string, and go string data.
	s = linklookup(ctxt, "type.*", 0);
	s->type = STYPE;
	s->size = 0;
	s->reachable = 1;
	symtype = s;

	s = linklookup(ctxt, "go.string.*", 0);
	s->type = SGOSTRING;
	s->size = 0;
	s->reachable = 1;
	symgostring = s;
	
	s = linklookup(ctxt, "go.func.*", 0);
	s->type = SGOFUNC;
	s->size = 0;
	s->reachable = 1;
	symgofunc = s;
	
	symtypelink = linklookup(ctxt, "typelink", 0);

	symt = linklookup(ctxt, "symtab", 0);
	symt->type = SSYMTAB;
	symt->size = 0;
	symt->reachable = 1;

	// assign specific types so that they sort together.
	// within a type they sort by size, so the .* symbols
	// just defined above will be first.
	// hide the specific symbols.
	for(s = ctxt->allsym; s != S; s = s->allsym) {
		if(!s->reachable || s->special || s->type != SRODATA)
			continue;
		if(strncmp(s->name, "type.", 5) == 0) {
			s->type = STYPE;
			s->hide = 1;
			s->outer = symtype;
		}
		if(strncmp(s->name, "go.typelink.", 12) == 0) {
			s->type = STYPELINK;
			s->hide = 1;
			s->outer = symtypelink;
		}
		if(strncmp(s->name, "go.string.", 10) == 0) {
			s->type = SGOSTRING;
			s->hide = 1;
			s->outer = symgostring;
		}
		if(strncmp(s->name, "go.func.", 8) == 0) {
			s->type = SGOFUNC;
			s->hide = 1;
			s->outer = symgofunc;
		}
	}
}

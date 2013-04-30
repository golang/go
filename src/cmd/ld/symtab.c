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
putelfsym(Sym *x, char *s, int t, vlong addr, vlong size, int ver, Sym *go)
{
	int bind, type, off;
	Sym *xo;

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
		cursym = x;
		diag("missing section in putelfsym");
		return;
	}
	if(xo->sect->elfsect == nil) {
		cursym = x;
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
putelfsectionsym(Sym* s, int shndx)
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
	Sym *s;

	// the first symbol entry is reserved
	putelfsyment(0, 0, 0, (STB_LOCAL<<4)|STT_NOTYPE, 0, 0);

	dwarfaddelfsectionsyms();

	elfbind = STB_LOCAL;
	genasmsym(putelfsym);
	
	if(linkmode == LinkExternal && HEADTYPE != Hopenbsd) {
		s = lookup("runtime.m", 0);
		if(s->sect == nil) {
			cursym = nil;
			diag("missing section for %s", s->name);
			errorexit();
		}
		putelfsyment(putelfstr(s->name), 0, PtrSize, (STB_LOCAL<<4)|STT_TLS, s->sect->elfsect->shnum, 0);
		s->elfsym = numelfsym++;

		s = lookup("runtime.g", 0);
		if(s->sect == nil) {
			cursym = nil;
			diag("missing section for %s", s->name);
			errorexit();
		}
		putelfsyment(putelfstr(s->name), PtrSize, PtrSize, (STB_LOCAL<<4)|STT_TLS, s->sect->elfsect->shnum, 0);
		s->elfsym = numelfsym++;
	}

	elfbind = STB_GLOBAL;
	elfglobalsymndx = numelfsym;
	genasmsym(putelfsym);
	
	for(s=allsym; s!=S; s=s->allsym) {
		if(s->type != SHOSTOBJ)
			continue;
		putelfsyment(putelfstr(s->name), 0, 0, (STB_GLOBAL<<4)|STT_NOTYPE, 0, 0);
		s->elfsym = numelfsym++;
	}
}

static void
putplan9sym(Sym *x, char *s, int t, vlong addr, vlong size, int ver, Sym *go)
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
		if(HEADTYPE == Hplan9x64 && !debug['8']) {
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

static Sym *symt;

static void
scput(int b)
{
	uchar *p;

	symgrow(symt, symt->size+1);
	p = symt->p + symt->size;
	*p = b;
	symt->size++;
}

static void
slputb(int32 v)
{
	uchar *p;

	symgrow(symt, symt->size+4);
	p = symt->p + symt->size;
	*p++ = v>>24;
	*p++ = v>>16;
	*p++ = v>>8;
	*p = v;
	symt->size += 4;
}

static void
slputl(int32 v)
{
	uchar *p;

	symgrow(symt, symt->size+4);
	p = symt->p + symt->size;
	*p++ = v;
	*p++ = v>>8;
	*p++ = v>>16;
	*p = v>>24;
	symt->size += 4;
}

static void (*slput)(int32);

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

// Emit symbol table entry.
// The table format is described at the top of ../../pkg/runtime/symtab.c.
void
putsymb(Sym *s, char *name, int t, vlong v, vlong size, int ver, Sym *typ)
{
	int i, f, c;
	vlong v1;
	Reloc *rel;

	USED(size);
	
	// type byte
	if('A' <= t && t <= 'Z')
		c = t - 'A' + (ver ? 26 : 0);
	else if('a' <= t && t <= 'z')
		c = t - 'a' + 26;
	else {
		diag("invalid symbol table type %c", t);
		errorexit();
		return;
	}
	
	if(s != nil)
		c |= 0x40; // wide value
	if(typ != nil)
		c |= 0x80; // has go type
	scput(c);

	// value
	if(s != nil) {
		// full width
		rel = addrel(symt);
		rel->siz = PtrSize;
		rel->sym = s;
		rel->type = D_ADDR;
		rel->off = symt->size;
		if(PtrSize == 8)
			slput(0);
		slput(0);
	} else {
		// varint
		if(v < 0) {
			diag("negative value in symbol table: %s %lld", name, v);
			errorexit();
		}
		v1 = v;
		while(v1 >= 0x80) {
			scput(v1 | 0x80);
			v1 >>= 7;
		}
		scput(v1);
	}

	// go type if present
	if(typ != nil) {
		if(!typ->reachable)
			diag("unreachable type %s", typ->name);
		rel = addrel(symt);
		rel->siz = PtrSize;
		rel->sym = typ;
		rel->type = D_ADDR;
		rel->off = symt->size;
		if(PtrSize == 8)
			slput(0);
		slput(0);
	}
	
	// name	
	if(t == 'f')
		name++;

	if(t == 'Z' || t == 'z') {
		scput(name[0]);
		for(i=1; name[i] != 0 || name[i+1] != 0; i += 2) {
			scput(name[i]);
			scput(name[i+1]);
		}
		scput(0);
		scput(0);
	} else {
		for(i=0; name[i]; i++)
			scput(name[i]);
		scput(0);
	}

	if(debug['n']) {
		if(t == 'z' || t == 'Z') {
			Bprint(&bso, "%c %.8llux ", t, v);
			for(i=1; name[i] != 0 || name[i+1] != 0; i+=2) {
				f = ((name[i]&0xff) << 8) | (name[i+1]&0xff);
				Bprint(&bso, "/%x", f);
			}
			Bprint(&bso, "\n");
			return;
		}
		if(ver)
			Bprint(&bso, "%c %.8llux %s<%d> %s\n", t, v, name, ver, typ ? typ->name : "");
		else
			Bprint(&bso, "%c %.8llux %s %s\n", t, v, name, typ ? typ->name : "");
	}
}

void
symtab(void)
{
	Sym *s, *symtype, *symtypelink, *symgostring;
	dosymtype();

	// Define these so that they'll get put into the symbol table.
	// data.c:/^address will provide the actual values.
	xdefine("text", STEXT, 0);
	xdefine("etext", STEXT, 0);
	xdefine("typelink", SRODATA, 0);
	xdefine("etypelink", SRODATA, 0);
	xdefine("rodata", SRODATA, 0);
	xdefine("erodata", SRODATA, 0);
	if(flag_shared) {
		xdefine("datarelro", SDATARELRO, 0);
		xdefine("edatarelro", SDATARELRO, 0);
	}
	xdefine("egcdata", STYPE, 0);
	xdefine("egcbss", STYPE, 0);
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

	// pseudo-symbols to mark locations of type, string, and go string data.
	s = lookup("type.*", 0);
	s->type = STYPE;
	s->size = 0;
	s->reachable = 1;
	symtype = s;

	s = lookup("go.string.*", 0);
	s->type = SGOSTRING;
	s->size = 0;
	s->reachable = 1;
	symgostring = s;
	
	symtypelink = lookup("typelink", 0);

	symt = lookup("symtab", 0);
	symt->type = SSYMTAB;
	symt->size = 0;
	symt->reachable = 1;

	// assign specific types so that they sort together.
	// within a type they sort by size, so the .* symbols
	// just defined above will be first.
	// hide the specific symbols.
	for(s = allsym; s != S; s = s->allsym) {
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
	}

	if(debug['s'])
		return;

	switch(thechar) {
	default:
		diag("unknown architecture %c", thechar);
		errorexit();
	case '5':
	case '6':
	case '8':
		// little-endian symbol table
		slput = slputl;
		break;
	case 'v':
		// big-endian symbol table
		slput = slputb;
		break;
	}
	// new symbol table header.
	slput(0xfffffffd);
	scput(0);
	scput(0);
	scput(0);
	scput(PtrSize);

	genasmsym(putsymb);
}

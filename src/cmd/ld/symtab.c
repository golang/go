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

static void
putelfsym(Sym *x, char *s, int t, vlong addr, vlong size, int ver, Sym *go)
{
	int bind, type, shndx, off;

	USED(go);
	switch(t) {
	default:
		return;
	case 'T':
		type = STT_FUNC;
		shndx = elftextsh + 0;
		break;
	case 'D':
		type = STT_OBJECT;
		if((x->type&SMASK) == SRODATA)
			shndx = elftextsh + 1;
		else
			shndx = elftextsh + 2;
		break;
	case 'B':
		type = STT_OBJECT;
		shndx = elftextsh + 3;
		break;
	}
	// TODO(minux): we need to place all STB_LOCAL precede all STB_GLOBAL and
	// STB_WEAK symbols in the symbol table
	bind = (ver || (x->type & SHIDDEN)) ? STB_LOCAL : STB_GLOBAL;
	off = putelfstr(s);
	putelfsyment(off, addr, size, (bind<<4)|(type&0xf), shndx, (x->type & SHIDDEN) ? 2 : 0);
}

void
asmelfsym(void)
{
	// the first symbol entry is reserved
	putelfsyment(0, 0, 0, (STB_LOCAL<<4)|STT_NOTYPE, 0, 0);
	genasmsym(putelfsym);
}

static void
putplan9sym(Sym *x, char *s, int t, vlong addr, vlong size, int ver, Sym *go)
{
	int i;

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
		symsize += 4 + 1 + i + 1;
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
putsymb(Sym *s, char *name, int t, vlong v, vlong size, int ver, Sym *typ)
{
	int i, f, l;
	Reloc *rel;

	USED(size);
	if(t == 'f')
		name++;
	l = 4;
//	if(!debug['8'])
//		l = 8;
	if(s != nil) {
		rel = addrel(symt);
		rel->siz = l + Rbig;
		rel->sym = s;
		rel->type = D_ADDR;
		rel->off = symt->size;
		v = 0;
	}	
	if(l == 8)
		slputb(v>>32);
	slputb(v);
	if(ver)
		t += 'a' - 'A';
	scput(t+0x80);			/* 0x80 is variable length */

	if(t == 'Z' || t == 'z') {
		scput(name[0]);
		for(i=1; name[i] != 0 || name[i+1] != 0; i += 2) {
			scput(name[i]);
			scput(name[i+1]);
		}
		scput(0);
		scput(0);
	}
	else {
		for(i=0; name[i]; i++)
			scput(name[i]);
		scput(0);
	}
	if(typ) {
		if(!typ->reachable)
			diag("unreachable type %s", typ->name);
		rel = addrel(symt);
		rel->siz = l;
		rel->sym = typ;
		rel->type = D_ADDR;
		rel->off = symt->size;
	}
	if(l == 8)
		slputb(0);
	slputb(0);

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
			Bprint(&bso, "%c %.8llux %s<%d> %s\n", t, v, s->name, ver, typ ? typ->name : "");
		else
			Bprint(&bso, "%c %.8llux %s %s\n", t, v, s->name, typ ? typ->name : "");
	}
}

void
symtab(void)
{
	Sym *s;

	dosymtype();

	// Define these so that they'll get put into the symbol table.
	// data.c:/^address will provide the actual values.
	xdefine("text", STEXT, 0);
	xdefine("etext", STEXT, 0);
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
	
	// pseudo-symbols to mark locations of type, string, and go string data.
	s = lookup("type.*", 0);
	s->type = STYPE;
	s->size = 0;
	s->reachable = 1;

	s = lookup("go.string.*", 0);
	s->type = SGOSTRING;
	s->size = 0;
	s->reachable = 1;

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
		}
		if(strncmp(s->name, "go.string.", 10) == 0) {
			s->type = SGOSTRING;
			s->hide = 1;
		}
	}

	if(debug['s'])
		return;
	genasmsym(putsymb);
}

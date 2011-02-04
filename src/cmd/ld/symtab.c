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

char *elfstrdat;
int elfstrsize;
int maxelfstr;
int elftextsh;

int
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

void
putelfsym64(Sym *x, char *s, int t, vlong addr, vlong size, int ver, Sym *go)
{
	int bind, type, shndx, stroff;
	
	bind = STB_GLOBAL;
	switch(t) {
	default:
		return;
	case 'T':
		type = STT_FUNC;
		shndx = elftextsh + 0;
		break;
	case 'D':
		type = STT_OBJECT;
		shndx = elftextsh + 1;
		break;
	case 'B':
		type = STT_OBJECT;
		shndx = elftextsh + 2;
		break;
	}
	
	stroff = putelfstr(s);
	LPUT(stroff);	// string
	cput((bind<<4)|(type&0xF));
	cput(0);
	WPUT(shndx);
	VPUT(addr);
	VPUT(size);
}

void
asmelfsym64(void)
{
	genasmsym(putelfsym64);
}

void
putelfsym32(Sym *x, char *s, int t, vlong addr, vlong size, int ver, Sym *go)
{
	int bind, type, shndx, stroff;
	
	bind = STB_GLOBAL;
	switch(t) {
	default:
		return;
	case 'T':
		type = STT_FUNC;
		shndx = elftextsh + 0;
		break;
	case 'D':
		type = STT_OBJECT;
		shndx = elftextsh + 1;
		break;
	case 'B':
		type = STT_OBJECT;
		shndx = elftextsh + 2;
		break;
	}
	
	stroff = putelfstr(s);
	LPUT(stroff);	// string
	LPUT(addr);
	LPUT(size);
	cput((bind<<4)|(type&0xF));
	cput(0);
	WPUT(shndx);
}

void
asmelfsym32(void)
{
	genasmsym(putelfsym32);
}

void
putplan9sym(Sym *x, char *s, int t, vlong addr, vlong size, int ver, Sym *go)
{
	int i;
		
	switch(t) {
	case 'T':
	case 't':
	case 'L':
	case 'l':
	case 'D':
	case 'd':
	case 'B':
	case 'b':
	case 'a':
	case 'p':
	
	case 'f':
	case 'z':
	case 'Z':
		
	case 'm':
		lputb(addr);
		cput(t+0x80); /* 0x80 is variable length */
		
		if(t == 'z' || t == 'Z') {
			cput(0);
			for(i=1; s[i] != 0 || s[i+1] != 0; i += 2) {
				cput(s[i]);
				cput(s[i+1]);
			}
			cput(0);
			cput(0);
			i++;
		} else {
			/* skip the '<' in filenames */
			if(t=='f')
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
putsymb(Sym *s, char *name, int t, vlong v, vlong size, int ver, Sym *typ)
{
	int i, f, l;
	Reloc *rel;

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
		i++;
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
			Bprint(&bso, "%c %.8llux %s<%d> %s\n", t, v, s, ver, typ ? typ->name : "");
		else
			Bprint(&bso, "%c %.8llux %s %s\n", t, v, s, typ ? typ->name : "");
	}
}

void
symtab(void)
{
	// Define these so that they'll get put into the symbol table.
	// data.c:/^address will provide the actual values.
	xdefine("text", STEXT, 0);
	xdefine("etext", STEXT, 0);
	xdefine("rodata", SRODATA, 0);
	xdefine("erodata", SRODATA, 0);
	xdefine("data", SBSS, 0);
	xdefine("edata", SBSS, 0);
	xdefine("end", SBSS, 0);
	xdefine("epclntab", SRODATA, 0);
	xdefine("esymtab", SRODATA, 0);

	symt = lookup("symtab", 0);
	symt->type = SRODATA;
	symt->size = 0;
	symt->reachable = 1;

	genasmsym(putsymb);
}

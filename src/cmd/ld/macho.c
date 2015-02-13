// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Mach-O file writing
// http://developer.apple.com/mac/library/DOCUMENTATION/DeveloperTools/Conceptual/MachORuntime/Reference/reference.html

#include <u.h>
#include <libc.h>
#include <bio.h>
#include <link.h>
#include "dwarf.h"
#include "lib.h"
#include "macho.h"

static	int	macho64;
static	MachoHdr	hdr;
static	MachoLoad	*load;
static	MachoSeg	seg[16];
static	int	nload, mload, nseg, ndebug, nsect;

enum
{
	SymKindLocal = 0,
	SymKindExtdef,
	SymKindUndef,
	NumSymKind
};

static	int nkind[NumSymKind];
static	LSym** sortsym;
static	int	nsortsym;

// Amount of space left for adding load commands
// that refer to dynamic libraries.  Because these have
// to go in the Mach-O header, we can't just pick a
// "big enough" header size.  The initial header is 
// one page, the non-dynamic library stuff takes
// up about 1300 bytes; we overestimate that as 2k.
static	int	load_budget = INITIAL_MACHO_HEADR - 2*1024;

static	void	machodysymtab(void);

void
machoinit(void)
{
	switch(thearch.thechar) {
	// 64-bit architectures
	case '6':
	case '9':
		macho64 = 1;
		break;

	// 32-bit architectures
	default:
		break;
	}
}

MachoHdr*
getMachoHdr(void)
{
	return &hdr;
}

MachoLoad*
newMachoLoad(uint32 type, uint32 ndata)
{
	MachoLoad *l;

	if(nload >= mload) {
		if(mload == 0)
			mload = 1;
		else
			mload *= 2;
		load = erealloc(load, mload*sizeof load[0]);
	}

	if(macho64 && (ndata & 1))
		ndata++;
	
	l = &load[nload++];
	l->type = type;
	l->ndata = ndata;
	l->data = mal(ndata*4);
	return l;
}

MachoSeg*
newMachoSeg(char *name, int msect)
{
	MachoSeg *s;

	if(nseg >= nelem(seg)) {
		diag("too many segs");
		errorexit();
	}
	s = &seg[nseg++];
	s->name = name;
	s->msect = msect;
	s->sect = mal(msect*sizeof s->sect[0]);
	return s;
}

MachoSect*
newMachoSect(MachoSeg *seg, char *name, char *segname)
{
	MachoSect *s;

	if(seg->nsect >= seg->msect) {
		diag("too many sects in segment %s", seg->name);
		errorexit();
	}
	s = &seg->sect[seg->nsect++];
	s->name = name;
	s->segname = segname;
	nsect++;
	return s;
}

// Generic linking code.

static char **dylib;
static int ndylib;

static vlong linkoff;

int
machowrite(void)
{
	vlong o1;
	int loadsize;
	int i, j;
	MachoSeg *s;
	MachoSect *t;
	MachoLoad *l;

	o1 = cpos();

	loadsize = 4*4*ndebug;
	for(i=0; i<nload; i++)
		loadsize += 4*(load[i].ndata+2);
	if(macho64) {
		loadsize += 18*4*nseg;
		loadsize += 20*4*nsect;
	} else {
		loadsize += 14*4*nseg;
		loadsize += 17*4*nsect;
	}

	if(macho64)
		thearch.lput(0xfeedfacf);
	else
		thearch.lput(0xfeedface);
	thearch.lput(hdr.cpu);
	thearch.lput(hdr.subcpu);
	if(linkmode == LinkExternal)
		thearch.lput(1);	/* file type - mach object */
	else
		thearch.lput(2);	/* file type - mach executable */
	thearch.lput(nload+nseg+ndebug);
	thearch.lput(loadsize);
	thearch.lput(1);	/* flags - no undefines */
	if(macho64)
		thearch.lput(0);	/* reserved */

	for(i=0; i<nseg; i++) {
		s = &seg[i];
		if(macho64) {
			thearch.lput(25);	/* segment 64 */
			thearch.lput(72+80*s->nsect);
			strnput(s->name, 16);
			thearch.vput(s->vaddr);
			thearch.vput(s->vsize);
			thearch.vput(s->fileoffset);
			thearch.vput(s->filesize);
			thearch.lput(s->prot1);
			thearch.lput(s->prot2);
			thearch.lput(s->nsect);
			thearch.lput(s->flag);
		} else {
			thearch.lput(1);	/* segment 32 */
			thearch.lput(56+68*s->nsect);
			strnput(s->name, 16);
			thearch.lput(s->vaddr);
			thearch.lput(s->vsize);
			thearch.lput(s->fileoffset);
			thearch.lput(s->filesize);
			thearch.lput(s->prot1);
			thearch.lput(s->prot2);
			thearch.lput(s->nsect);
			thearch.lput(s->flag);
		}
		for(j=0; j<s->nsect; j++) {
			t = &s->sect[j];
			if(macho64) {
				strnput(t->name, 16);
				strnput(t->segname, 16);
				thearch.vput(t->addr);
				thearch.vput(t->size);
				thearch.lput(t->off);
				thearch.lput(t->align);
				thearch.lput(t->reloc);
				thearch.lput(t->nreloc);
				thearch.lput(t->flag);
				thearch.lput(t->res1);	/* reserved */
				thearch.lput(t->res2);	/* reserved */
				thearch.lput(0);	/* reserved */
			} else {
				strnput(t->name, 16);
				strnput(t->segname, 16);
				thearch.lput(t->addr);
				thearch.lput(t->size);
				thearch.lput(t->off);
				thearch.lput(t->align);
				thearch.lput(t->reloc);
				thearch.lput(t->nreloc);
				thearch.lput(t->flag);
				thearch.lput(t->res1);	/* reserved */
				thearch.lput(t->res2);	/* reserved */
			}
		}
	}

	for(i=0; i<nload; i++) {
		l = &load[i];
		thearch.lput(l->type);
		thearch.lput(4*(l->ndata+2));
		for(j=0; j<l->ndata; j++)
			thearch.lput(l->data[j]);
	}

	return cpos() - o1;
}

void
domacho(void)
{
	LSym *s;

	if(debug['d'])
		return;

	// empirically, string table must begin with " \x00".
	s = linklookup(ctxt, ".machosymstr", 0);
	s->type = SMACHOSYMSTR;
	s->reachable = 1;
	adduint8(ctxt, s, ' ');
	adduint8(ctxt, s, '\0');
	
	s = linklookup(ctxt, ".machosymtab", 0);
	s->type = SMACHOSYMTAB;
	s->reachable = 1;
	
	if(linkmode != LinkExternal) {
		s = linklookup(ctxt, ".plt", 0);	// will be __symbol_stub
		s->type = SMACHOPLT;
		s->reachable = 1;
	
		s = linklookup(ctxt, ".got", 0);	// will be __nl_symbol_ptr
		s->type = SMACHOGOT;
		s->reachable = 1;
		s->align = 4;
	
		s = linklookup(ctxt, ".linkedit.plt", 0);	// indirect table for .plt
		s->type = SMACHOINDIRECTPLT;
		s->reachable = 1;
	
		s = linklookup(ctxt, ".linkedit.got", 0);	// indirect table for .got
		s->type = SMACHOINDIRECTGOT;
		s->reachable = 1;
	}
}

void
machoadddynlib(char *lib)
{
	// Will need to store the library name rounded up
	// and 24 bytes of header metadata.  If not enough
	// space, grab another page of initial space at the
	// beginning of the output file.
	load_budget -= (strlen(lib)+7)/8*8 + 24;
	if(load_budget < 0) {
		HEADR += 4096;
		INITTEXT += 4096;
		load_budget += 4096;
	}

	if(ndylib%32 == 0)
		dylib = erealloc(dylib, (ndylib+32)*sizeof dylib[0]);
	dylib[ndylib++] = lib;
}

static void
machoshbits(MachoSeg *mseg, Section *sect, char *segname)
{
	MachoSect *msect;
	char buf[40];
	char *p;
	
	snprint(buf, sizeof buf, "__%s", sect->name+1);
	for(p=buf; *p; p++)
		if(*p == '.')
			*p = '_';

	msect = newMachoSect(mseg, estrdup(buf), segname);
	if(sect->rellen > 0) {
		msect->reloc = sect->reloff;
		msect->nreloc = sect->rellen / 8;
	}

	while(1<<msect->align < sect->align)
		msect->align++;
	msect->addr = sect->vaddr;
	msect->size = sect->len;
	
	if(sect->vaddr < sect->seg->vaddr + sect->seg->filelen) {
		// data in file
		if(sect->len > sect->seg->vaddr + sect->seg->filelen - sect->vaddr)
			diag("macho cannot represent section %s crossing data and bss", sect->name);
		msect->off = sect->seg->fileoff + sect->vaddr - sect->seg->vaddr;
	} else {
		// zero fill
		msect->off = 0;
		msect->flag |= 1;
	}

	if(sect->rwx & 1)
		msect->flag |= 0x400; /* has instructions */
	
	if(strcmp(sect->name, ".plt") == 0) {
		msect->name = "__symbol_stub1";
		msect->flag = 0x80000408; /* only instructions, code, symbol stubs */
		msect->res1 = 0;//nkind[SymKindLocal];
		msect->res2 = 6;
	}

	if(strcmp(sect->name, ".got") == 0) {
		msect->name = "__nl_symbol_ptr";
		msect->flag = 6;	/* section with nonlazy symbol pointers */
		msect->res1 = linklookup(ctxt, ".linkedit.plt", 0)->size / 4;	/* offset into indirect symbol table */
	}
}

void
asmbmacho(void)
{
	vlong v, w;
	vlong va;
	int a, i;
	MachoHdr *mh;
	MachoSeg *ms;
	MachoLoad *ml;
	Section *sect;

	/* apple MACH */
	va = INITTEXT - HEADR;
	mh = getMachoHdr();
	switch(thearch.thechar){
	default:
		diag("unknown mach architecture");
		errorexit();
	case '5':
		mh->cpu = MACHO_CPU_ARM;
		mh->subcpu = MACHO_SUBCPU_ARMV7;
		break;
	case '6':
		mh->cpu = MACHO_CPU_AMD64;
		mh->subcpu = MACHO_SUBCPU_X86;
		break;
	case '8':
		mh->cpu = MACHO_CPU_386;
		mh->subcpu = MACHO_SUBCPU_X86;
		break;
	}
	
	ms = nil;
	if(linkmode == LinkExternal) {
		/* segment for entire file */
		ms = newMachoSeg("", 40);
		ms->fileoffset = segtext.fileoff;
		ms->filesize = segdata.fileoff + segdata.filelen - segtext.fileoff;
	}

	/* segment for zero page */
	if(linkmode != LinkExternal) {
		ms = newMachoSeg("__PAGEZERO", 0);
		ms->vsize = va;
	}

	/* text */
	v = rnd(HEADR+segtext.len, INITRND);
	if(linkmode != LinkExternal) {
		ms = newMachoSeg("__TEXT", 20);
		ms->vaddr = va;
		ms->vsize = v;
		ms->fileoffset = 0;
		ms->filesize = v;
		ms->prot1 = 7;
		ms->prot2 = 5;
	}

	for(sect=segtext.sect; sect!=nil; sect=sect->next)
		machoshbits(ms, sect, "__TEXT");

	/* data */
	if(linkmode != LinkExternal) {
		w = segdata.len;
		ms = newMachoSeg("__DATA", 20);
		ms->vaddr = va+v;
		ms->vsize = w;
		ms->fileoffset = v;
		ms->filesize = segdata.filelen;
		ms->prot1 = 3;
		ms->prot2 = 3;
	}

	for(sect=segdata.sect; sect!=nil; sect=sect->next)
		machoshbits(ms, sect, "__DATA");

	if(linkmode != LinkExternal) {
		switch(thearch.thechar) {
		default:
			diag("unknown macho architecture");
			errorexit();
		case '5':
			ml = newMachoLoad(5, 17+2);	/* unix thread */
			ml->data[0] = 1;	/* thread type */
			ml->data[1] = 17;	/* word count */
			ml->data[2+15] = entryvalue();	/* start pc */
			break;
		case '6':
			ml = newMachoLoad(5, 42+2);	/* unix thread */
			ml->data[0] = 4;	/* thread type */
			ml->data[1] = 42;	/* word count */
			ml->data[2+32] = entryvalue();	/* start pc */
			ml->data[2+32+1] = entryvalue()>>16>>16;	// hide >>32 for 8l
			break;
		case '8':
			ml = newMachoLoad(5, 16+2);	/* unix thread */
			ml->data[0] = 1;	/* thread type */
			ml->data[1] = 16;	/* word count */
			ml->data[2+10] = entryvalue();	/* start pc */
			break;
		}
	}
	
	if(!debug['d']) {
		LSym *s1, *s2, *s3, *s4;

		// must match domacholink below
		s1 = linklookup(ctxt, ".machosymtab", 0);
		s2 = linklookup(ctxt, ".linkedit.plt", 0);
		s3 = linklookup(ctxt, ".linkedit.got", 0);
		s4 = linklookup(ctxt, ".machosymstr", 0);

		if(linkmode != LinkExternal) {
			ms = newMachoSeg("__LINKEDIT", 0);
			ms->vaddr = va+v+rnd(segdata.len, INITRND);
			ms->vsize = s1->size + s2->size + s3->size + s4->size;
			ms->fileoffset = linkoff;
			ms->filesize = ms->vsize;
			ms->prot1 = 7;
			ms->prot2 = 3;
		}

		ml = newMachoLoad(2, 4);	/* LC_SYMTAB */
		ml->data[0] = linkoff;	/* symoff */
		ml->data[1] = nsortsym;	/* nsyms */
		ml->data[2] = linkoff + s1->size + s2->size + s3->size;	/* stroff */
		ml->data[3] = s4->size;	/* strsize */

		machodysymtab();

		if(linkmode != LinkExternal) {
			ml = newMachoLoad(14, 6);	/* LC_LOAD_DYLINKER */
			ml->data[0] = 12;	/* offset to string */
			strcpy((char*)&ml->data[1], "/usr/lib/dyld");
	
			for(i=0; i<ndylib; i++) {
				ml = newMachoLoad(12, 4+(strlen(dylib[i])+1+7)/8*2);	/* LC_LOAD_DYLIB */
				ml->data[0] = 24;	/* offset of string from beginning of load */
				ml->data[1] = 0;	/* time stamp */
				ml->data[2] = 0;	/* version */
				ml->data[3] = 0;	/* compatibility version */
				strcpy((char*)&ml->data[4], dylib[i]);
			}
		}
	}

	// TODO: dwarf headers go in ms too
	if(!debug['s'] && linkmode != LinkExternal)
		dwarfaddmachoheaders();

	a = machowrite();
	if(a > HEADR)
		diag("HEADR too small: %d > %d", a, HEADR);
}

static int
symkind(LSym *s)
{
	if(s->type == SDYNIMPORT)
		return SymKindUndef;
	if(s->cgoexport)
		return SymKindExtdef;
	return SymKindLocal;
}

static void
addsym(LSym *s, char *name, int type, vlong addr, vlong size, int ver, LSym *gotype)
{
	USED(name);
	USED(addr);
	USED(size);
	USED(ver);
	USED(gotype);

	if(s == nil)
		return;

	switch(type) {
	default:
		return;
	case 'D':
	case 'B':
	case 'T':
		break;
	}
	
	if(sortsym) {
		sortsym[nsortsym] = s;
		nkind[symkind(s)]++;
	}
	nsortsym++;
}
	
static int
scmp(const void *p1, const void *p2)
{
	LSym *s1, *s2;
	int k1, k2;

	s1 = *(LSym**)p1;
	s2 = *(LSym**)p2;
	
	k1 = symkind(s1);
	k2 = symkind(s2);
	if(k1 != k2)
		return k1 - k2;

	return strcmp(s1->extname, s2->extname);
}

static void
machogenasmsym(void (*put)(LSym*, char*, int, vlong, vlong, int, LSym*))
{
	LSym *s;

	genasmsym(put);
	for(s=ctxt->allsym; s; s=s->allsym)
		if(s->type == SDYNIMPORT || s->type == SHOSTOBJ)
		if(s->reachable)
			put(s, nil, 'D', 0, 0, 0, nil);
}
			
void
machosymorder(void)
{
	int i;

	// On Mac OS X Mountain Lion, we must sort exported symbols
	// So we sort them here and pre-allocate dynid for them
	// See http://golang.org/issue/4029
	for(i=0; i<ndynexp; i++)
		dynexp[i]->reachable = 1;
	machogenasmsym(addsym);
	sortsym = mal(nsortsym * sizeof sortsym[0]);
	nsortsym = 0;
	machogenasmsym(addsym);
	qsort(sortsym, nsortsym, sizeof sortsym[0], scmp);
	for(i=0; i<nsortsym; i++)
		sortsym[i]->dynid = i;
}

static void
machosymtab(void)
{
	int i;
	LSym *symtab, *symstr, *s, *o;
	char *p;

	symtab = linklookup(ctxt, ".machosymtab", 0);
	symstr = linklookup(ctxt, ".machosymstr", 0);

	for(i=0; i<nsortsym; i++) {
		s = sortsym[i];
		adduint32(ctxt, symtab, symstr->size);
		
		// Only add _ to C symbols. Go symbols have dot in the name.
		if(strstr(s->extname, ".") == nil)
			adduint8(ctxt, symstr, '_');
		// replace "·" as ".", because DTrace cannot handle it.
		if(strstr(s->extname, "·") == nil) {
			addstring(symstr, s->extname);
		} else {
			for(p = s->extname; *p; p++) {
				if((uchar)*p == 0xc2 && (uchar)*(p+1) == 0xb7) {
					adduint8(ctxt, symstr, '.');
					p++;
				} else {
					adduint8(ctxt, symstr, *p);
				}
			}
			adduint8(ctxt, symstr, '\0');
		}
		if(s->type == SDYNIMPORT || s->type == SHOSTOBJ) {
			adduint8(ctxt, symtab, 0x01); // type N_EXT, external symbol
			adduint8(ctxt, symtab, 0); // no section
			adduint16(ctxt, symtab, 0); // desc
			adduintxx(ctxt, symtab, 0, thearch.ptrsize); // no value
		} else {
			if(s->cgoexport)
				adduint8(ctxt, symtab, 0x0f);
			else
				adduint8(ctxt, symtab, 0x0e);
			o = s;
			while(o->outer != nil)
				o = o->outer;
			if(o->sect == nil) {
				diag("missing section for %s", s->name);
				adduint8(ctxt, symtab, 0);
			} else
				adduint8(ctxt, symtab, o->sect->extnum);
			adduint16(ctxt, symtab, 0); // desc
			adduintxx(ctxt, symtab, symaddr(s), thearch.ptrsize);
		}
	}
}

static void
machodysymtab(void)
{
	int n;
	MachoLoad *ml;
	LSym *s1, *s2, *s3;

	ml = newMachoLoad(11, 18);	/* LC_DYSYMTAB */

	n = 0;
	ml->data[0] = n;	/* ilocalsym */
	ml->data[1] = nkind[SymKindLocal];	/* nlocalsym */
	n += nkind[SymKindLocal];

	ml->data[2] = n;	/* iextdefsym */
	ml->data[3] = nkind[SymKindExtdef];	/* nextdefsym */
	n += nkind[SymKindExtdef];

	ml->data[4] = n;	/* iundefsym */
	ml->data[5] = nkind[SymKindUndef];	/* nundefsym */

	ml->data[6] = 0;	/* tocoffset */
	ml->data[7] = 0;	/* ntoc */
	ml->data[8] = 0;	/* modtaboff */
	ml->data[9] = 0;	/* nmodtab */
	ml->data[10] = 0;	/* extrefsymoff */
	ml->data[11] = 0;	/* nextrefsyms */

	// must match domacholink below
	s1 = linklookup(ctxt, ".machosymtab", 0);
	s2 = linklookup(ctxt, ".linkedit.plt", 0);
	s3 = linklookup(ctxt, ".linkedit.got", 0);
	ml->data[12] = linkoff + s1->size;	/* indirectsymoff */
	ml->data[13] = (s2->size + s3->size) / 4;	/* nindirectsyms */

	ml->data[14] = 0;	/* extreloff */
	ml->data[15] = 0;	/* nextrel */
	ml->data[16] = 0;	/* locreloff */
	ml->data[17] = 0;	/* nlocrel */
}

vlong
domacholink(void)
{
	int size;
	LSym *s1, *s2, *s3, *s4;

	machosymtab();

	// write data that will be linkedit section
	s1 = linklookup(ctxt, ".machosymtab", 0);
	s2 = linklookup(ctxt, ".linkedit.plt", 0);
	s3 = linklookup(ctxt, ".linkedit.got", 0);
	s4 = linklookup(ctxt, ".machosymstr", 0);

	// Force the linkedit section to end on a 16-byte
	// boundary.  This allows pure (non-cgo) Go binaries
	// to be code signed correctly.
	//
	// Apple's codesign_allocate (a helper utility for
	// the codesign utility) can do this fine itself if
	// it is run on a dynamic Mach-O binary.  However,
	// when it is run on a pure (non-cgo) Go binary, where
	// the linkedit section is mostly empty, it fails to
	// account for the extra padding that it itself adds
	// when adding the LC_CODE_SIGNATURE load command
	// (which must be aligned on a 16-byte boundary).
	//
	// By forcing the linkedit section to end on a 16-byte
	// boundary, codesign_allocate will not need to apply
	// any alignment padding itself, working around the
	// issue.
	while(s4->size%16)
		adduint8(ctxt, s4, 0);
	
	size = s1->size + s2->size + s3->size + s4->size;

	if(size > 0) {
		linkoff = rnd(HEADR+segtext.len, INITRND) + rnd(segdata.filelen, INITRND) + rnd(segdwarf.filelen, INITRND);
		cseek(linkoff);

		cwrite(s1->p, s1->size);
		cwrite(s2->p, s2->size);
		cwrite(s3->p, s3->size);
		cwrite(s4->p, s4->size);
	}

	return rnd(size, INITRND);
}


void
machorelocsect(Section *sect, LSym *first)
{
	LSym *sym;
	int32 eaddr;
	Reloc *r;

	// If main section has no bits, nothing to relocate.
	if(sect->vaddr >= sect->seg->vaddr + sect->seg->filelen)
		return;
	
	sect->reloff = cpos();
	for(sym = first; sym != nil; sym = sym->next) {
		if(!sym->reachable)
			continue;
		if(sym->value >= sect->vaddr)
			break;
	}
	
	eaddr = sect->vaddr + sect->len;
	for(; sym != nil; sym = sym->next) {
		if(!sym->reachable)
			continue;
		if(sym->value >= eaddr)
			break;
		ctxt->cursym = sym;
		
		for(r = sym->r; r < sym->r+sym->nr; r++) {
			if(r->done)
				continue;
			if(thearch.machoreloc1(r, sym->value+r->off - sect->vaddr) < 0)
				diag("unsupported obj reloc %d/%d to %s", r->type, r->siz, r->sym->name);
		}
	}
		
	sect->rellen = cpos() - sect->reloff;
}

void
machoemitreloc(void)
{
	Section *sect;

	while(cpos()&7)
		cput(0);

	machorelocsect(segtext.sect, ctxt->textp);
	for(sect=segtext.sect->next; sect!=nil; sect=sect->next)
		machorelocsect(sect, datap);	
	for(sect=segdata.sect; sect!=nil; sect=sect->next)
		machorelocsect(sect, datap);	
}

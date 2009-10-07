// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Mach-O file writing
// http://developer.apple.com/mac/library/DOCUMENTATION/DeveloperTools/Conceptual/MachORuntime/Reference/reference.html

#include "l.h"
#include "../ld/lib.h"
#include "../ld/macho.h"

static	int	macho64;
static	MachoHdr	hdr;
static	MachoLoad	load[16];
static	MachoSeg	seg[16];
static	MachoDebug	xdebug[16];
static	int	nload, nseg, ndebug, nsect;

void
machoinit(void)
{
	switch(thechar) {
	// 64-bit architectures
	case '6':
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

	if(nload >= nelem(load)) {
		diag("too many loads");
		errorexit();
	}
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
newMachoSect(MachoSeg *seg, char *name)
{
	MachoSect *s;

	if(seg->nsect >= seg->msect) {
		diag("too many sects in segment %s", seg->name);
		errorexit();
	}
	s = &seg->sect[seg->nsect++];
	s->name = name;
	nsect++;
	return s;
}

MachoDebug*
newMachoDebug(void)
{
	if(ndebug >= nelem(xdebug)) {
		diag("too many debugs");
		errorexit();
	}
	return &xdebug[ndebug++];
}


// Generic linking code.

static uchar *linkdata;
static uint32 nlinkdata;
static uint32 mlinkdata;

static uchar *strtab;
static uint32 nstrtab;
static uint32 mstrtab;

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
	MachoDebug *d;
	MachoLoad *l;

	o1 = Boffset(&bso);

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
		LPUT(0xfeedfacf);
	else
		LPUT(0xfeedface);
	LPUT(hdr.cpu);
	LPUT(hdr.subcpu);
	LPUT(2);	/* file type - mach executable */
	LPUT(nload+nseg+ndebug);
	LPUT(loadsize);
	LPUT(1);	/* flags - no undefines */
	if(macho64)
		LPUT(0);	/* reserved */

	for(i=0; i<nseg; i++) {
		s = &seg[i];
		if(macho64) {
			LPUT(25);	/* segment 64 */
			LPUT(72+80*s->nsect);
			strnput(s->name, 16);
			VPUT(s->vaddr);
			VPUT(s->vsize);
			VPUT(s->fileoffset);
			VPUT(s->filesize);
			LPUT(s->prot1);
			LPUT(s->prot2);
			LPUT(s->nsect);
			LPUT(s->flag);
		} else {
			LPUT(1);	/* segment 32 */
			LPUT(56+68*s->nsect);
			strnput(s->name, 16);
			LPUT(s->vaddr);
			LPUT(s->vsize);
			LPUT(s->fileoffset);
			LPUT(s->filesize);
			LPUT(s->prot1);
			LPUT(s->prot2);
			LPUT(s->nsect);
			LPUT(s->flag);
		}
		for(j=0; j<s->nsect; j++) {
			t = &s->sect[j];
			if(macho64) {
				strnput(t->name, 16);
				strnput(s->name, 16);
				VPUT(t->addr);
				VPUT(t->size);
				LPUT(t->off);
				LPUT(t->align);
				LPUT(t->reloc);
				LPUT(t->nreloc);
				LPUT(t->flag);
				LPUT(0);	/* reserved */
				LPUT(0);	/* reserved */
				LPUT(0);	/* reserved */
			} else {
				strnput(t->name, 16);
				strnput(s->name, 16);
				LPUT(t->addr);
				LPUT(t->size);
				LPUT(t->off);
				LPUT(t->align);
				LPUT(t->reloc);
				LPUT(t->nreloc);
				LPUT(t->flag);
				LPUT(0);	/* reserved */
				LPUT(0);	/* reserved */
			}
		}
	}

	for(i=0; i<nload; i++) {
		l = &load[i];
		LPUT(l->type);
		LPUT(4*(l->ndata+2));
		for(j=0; j<l->ndata; j++)
			LPUT(l->data[j]);
	}

	for(i=0; i<ndebug; i++) {
		d = &xdebug[i];
		LPUT(3);	/* obsolete gdb debug info */
		LPUT(16);	/* size of symseg command */
		LPUT(d->fileoffset);
		LPUT(d->filesize);
	}

	return Boffset(&bso) - o1;
}

static void*
grow(uchar **dat, uint32 *ndat, uint32 *mdat, uint32 n)
{
	uchar *p;
	uint32 old;

	if(*ndat+n > *mdat) {
		old = *mdat;
		*mdat = (*ndat+n)*2 + 128;
		*dat = realloc(*dat, *mdat);
		if(*dat == 0) {
			diag("out of memory");
			errorexit();
		}
		memset(*dat+old, 0, *mdat-old);
	}
	p = *dat + *ndat;
	*ndat += n;
	return p;
}

static int
needlib(char *name)
{
	char *p;
	Sym *s;

	/* reuse hash code in symbol table */
	p = smprint(".machoload.%s", name);
	s = lookup(p, 0);
	if(s->type == 0) {
		s->type = 100;	// avoid SDATA, etc.
		return 1;
	}
	return 0;
}

void
domacho(void)
{
	int h, nsym, ptrsize;
	char *p;
	uchar *dat;
	uint32 x;
	Sym *s;

	ptrsize = 4;
	if(macho64)
		ptrsize = 8;

	// empirically, string table must begin with " \x00".
	if(!debug['d'])
		*(char*)grow(&strtab, &nstrtab, &mstrtab, 2) = ' ';

	nsym = 0;
	for(h=0; h<NHASH; h++) {
		for(s=hash[h]; s!=S; s=s->link) {
			if(!s->reachable || (s->type != SDATA && s->type != SBSS) || s->dynldname == nil)
				continue;
			if(debug['d']) {
				diag("cannot use dynamic loading and -d");
				errorexit();
			}
			s->type = SMACHO;
			s->value = nsym*ptrsize;

			/* symbol table entry - darwin still puts _ prefixes on all C symbols */
			x = nstrtab;
			p = grow(&strtab, &nstrtab, &mstrtab, 1+strlen(s->dynldname)+1);
			*p++ = '_';
			strcpy(p, s->dynldname);

			dat = grow(&linkdata, &nlinkdata, &mlinkdata, 8+ptrsize);
			dat[0] = x;
			dat[1] = x>>8;
			dat[2] = x>>16;
			dat[3] = x>>24;
			dat[4] = 0x01;	// type: N_EXT - external symbol

			if(needlib(s->dynldlib)) {
				if(ndylib%32 == 0) {
					dylib = realloc(dylib, (ndylib+32)*sizeof dylib[0]);
					if(dylib == nil) {
						diag("out of memory");
						errorexit();
					}
				}
				dylib[ndylib++] = s->dynldlib;
			}
			nsym++;
		}
	}

	/*
	 * list of symbol table indexes.
	 * we don't take advantage of the opportunity
	 * to order the symbol table differently from
	 * this list, so it is boring: 0 1 2 3 4 ...
	 */
	for(x=0; x<nsym; x++) {
		dat = grow(&linkdata, &nlinkdata, &mlinkdata, 4);
		dat[0] = x;
		dat[1] = x>>8;
		dat[2] = x>>16;
		dat[3] = x>>24;
	}

	dynptrsize = nsym*ptrsize;
}

vlong
domacholink(void)
{
	linkoff = 0;
	if(nlinkdata > 0) {
		linkoff = rnd(HEADR+textsize, INITRND) + rnd(datsize, INITRND);
		seek(cout, linkoff, 0);
		write(cout, linkdata, nlinkdata);
		write(cout, strtab, nstrtab);
	}
	return rnd(nlinkdata+nstrtab, INITRND);
}

void
asmbmacho(vlong symdatva, vlong symo)
{
	vlong v, w;
	vlong va;
	int a, i, ptrsize;
	MachoHdr *mh;
	MachoSect *msect;
	MachoSeg *ms;
	MachoDebug *md;
	MachoLoad *ml;

	/* apple MACH */
	va = INITTEXT - HEADR;
	mh = getMachoHdr();
	switch(thechar){
	default:
		diag("unknown mach architecture");
		errorexit();
	case '6':
		mh->cpu = MACHO_CPU_AMD64;
		mh->subcpu = MACHO_SUBCPU_X86;
		ptrsize = 8;
		break;
	case '8':
		mh->cpu = MACHO_CPU_386;
		mh->subcpu = MACHO_SUBCPU_X86;
		ptrsize = 4;
		break;
	}

	/* segment for zero page */
	ms = newMachoSeg("__PAGEZERO", 0);
	ms->vsize = va;

	/* text */
	v = rnd(HEADR+textsize, INITRND);
	ms = newMachoSeg("__TEXT", 1);
	ms->vaddr = va;
	ms->vsize = v;
	ms->filesize = v;
	ms->prot1 = 7;
	ms->prot2 = 5;

	msect = newMachoSect(ms, "__text");
	msect->addr = INITTEXT;
	msect->size = textsize;
	msect->off = INITTEXT - va;
	msect->flag = 0x400;	/* flag - some instructions */

	/* data */
	w = datsize+dynptrsize+bsssize;
	ms = newMachoSeg("__DATA", 2+(dynptrsize>0));
	ms->vaddr = va+v;
	ms->vsize = w;
	ms->fileoffset = v;
	ms->filesize = datsize;
	ms->prot1 = 7;
	ms->prot2 = 3;

	msect = newMachoSect(ms, "__data");
	msect->addr = va+v;
	msect->size = datsize;
	msect->off = v;

	if(dynptrsize > 0) {
		msect = newMachoSect(ms, "__nl_symbol_ptr");
		msect->addr = va+v+datsize;
		msect->size = dynptrsize;
		msect->align = 2;
		msect->flag = 6;	/* section with nonlazy symbol pointers */
		/*
		 * The reserved1 field is supposed to be the index of
		 * the first entry in the list of symbol table indexes
		 * in isymtab for the symbols we need.  We only use
		 * pointers, so we need the entire list, so the index
		 * here should be 0, which luckily is what the Mach-O
		 * writing code emits by default for this not really reserved field.
		msect->reserved1 = 0; - first indirect symbol table entry we need
		 */
	}

	msect = newMachoSect(ms, "__bss");
	msect->addr = va+v+datsize+dynptrsize;
	msect->size = bsssize;
	msect->flag = 1;	/* flag - zero fill */

	switch(thechar) {
	default:
		diag("unknown macho architecture");
		errorexit();
	case '6':
		ml = newMachoLoad(5, 42+2);	/* unix thread */
		ml->data[0] = 4;	/* thread type */
		ml->data[1] = 42;	/* word count */
		ml->data[2+32] = entryvalue();	/* start pc */
		ml->data[2+32+1] = entryvalue()>>32;
		break;
	case '8':
		ml = newMachoLoad(5, 16+2);	/* unix thread */
		ml->data[0] = 1;	/* thread type */
		ml->data[1] = 16;	/* word count */
		ml->data[2+10] = entryvalue();	/* start pc */
		break;
	}

	if(!debug['d']) {
		int nsym;

		nsym = dynptrsize/ptrsize;

		ms = newMachoSeg("__LINKEDIT", 0);
		ms->vaddr = va+v+rnd(datsize+dynptrsize+bsssize, INITRND);
		ms->vsize = nlinkdata+nstrtab;
		ms->fileoffset = linkoff;
		ms->filesize = nlinkdata+nstrtab;
		ms->prot1 = 7;
		ms->prot2 = 3;

		ml = newMachoLoad(2, 4);	/* LC_SYMTAB */
		ml->data[0] = linkoff;	/* symoff */
		ml->data[1] = nsym;	/* nsyms */
		ml->data[2] = linkoff + nlinkdata;	/* stroff */
		ml->data[3] = nstrtab;	/* strsize */

		ml = newMachoLoad(11, 18);	/* LC_DYSYMTAB */
		ml->data[0] = 0;	/* ilocalsym */
		ml->data[1] = 0;	/* nlocalsym */
		ml->data[2] = 0;	/* iextdefsym */
		ml->data[3] = 0;	/* nextdefsym */
		ml->data[4] = 0;	/* iundefsym */
		ml->data[5] = nsym;	/* nundefsym */
		ml->data[6] = 0;	/* tocoffset */
		ml->data[7] = 0;	/* ntoc */
		ml->data[8] = 0;	/* modtaboff */
		ml->data[9] = 0;	/* nmodtab */
		ml->data[10] = 0;	/* extrefsymoff */
		ml->data[11] = 0;	/* nextrefsyms */
		ml->data[12] = linkoff + nlinkdata - nsym*4;	/* indirectsymoff */
		ml->data[13] = nsym;	/* nindirectsyms */
		ml->data[14] = 0;	/* extreloff */
		ml->data[15] = 0;	/* nextrel */
		ml->data[16] = 0;	/* locreloff */
		ml->data[17] = 0;	/* nlocrel */

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

	if(!debug['s']) {
		ms = newMachoSeg("__SYMDAT", 1);
		ms->vaddr = symdatva;
		ms->vsize = 8+symsize+lcsize;
		ms->fileoffset = symo;
		ms->filesize = 8+symsize+lcsize;
		ms->prot1 = 7;
		ms->prot2 = 5;

		md = newMachoDebug();
		md->fileoffset = symo+8;
		md->filesize = symsize;

		md = newMachoDebug();
		md->fileoffset = symo+8+symsize;
		md->filesize = lcsize;
	}

	a = machowrite();
	if(a > MACHORESERVE)
		diag("MACHORESERVE too small: %d > %d", a, MACHORESERVE);
}

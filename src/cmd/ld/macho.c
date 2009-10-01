// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Mach-O file writing
// http://developer.apple.com/mac/library/DOCUMENTATION/DeveloperTools/Conceptual/MachORuntime/Reference/reference.html

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

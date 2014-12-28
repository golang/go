// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"l.h"
#include	"lib.h"
#include	"../../runtime/funcdata.h"

static void
addvarint(Pcdata *d, uint32 val)
{
	int32 n;
	uint32 v;
	uchar *p;

	n = 0;
	for(v = val; v >= 0x80; v >>= 7)
		n++;
	n++;

	if(d->n + n > d->m) {
		d->m = (d->n + n)*2;
		d->p = erealloc(d->p, d->m);
	}

	p = d->p + d->n;
	for(v = val; v >= 0x80; v >>= 7)
		*p++ = v | 0x80;
	*p = v;
	d->n += n;
}

static int32
addpctab(LSym *ftab, int32 off, Pcdata *d)
{
	int32 start;
	
	start = ftab->np;
	symgrow(ctxt, ftab, start + d->n);
	memmove(ftab->p + start, d->p, d->n);
	
	return setuint32(ctxt, ftab, off, start);
}

static int32
ftabaddstring(LSym *ftab, char *s)
{
	int32 n, start;
	
	n = strlen(s)+1;
	start = ftab->np;
	symgrow(ctxt, ftab, start+n+1);
	strcpy((char*)ftab->p + start, s);
	return start;
}

static void
renumberfiles(Link *ctxt, LSym **files, int nfiles, Pcdata *d)
{
	int i;
	LSym *f;
	Pcdata out;
	Pciter it;
	uint32 v;
	int32 oldval, newval, val, dv;
	
	// Give files numbers.
	for(i=0; i<nfiles; i++) {
		f = files[i];
		if(f->type != SFILEPATH) {
			f->value = ++ctxt->nhistfile;
			f->type = SFILEPATH;
			f->next = ctxt->filesyms;
			ctxt->filesyms = f;
		}
	}

	newval = -1;
	memset(&out, 0, sizeof out);

	for(pciterinit(ctxt, &it, d); !it.done; pciternext(&it)) {
		// value delta
		oldval = it.value;
		if(oldval == -1)
			val = -1;
		else {	
			if(oldval < 0 || oldval >= nfiles)
				sysfatal("bad pcdata %d", oldval);
			val = files[oldval]->value;
		}
		dv = val - newval;
		newval = val;
		v = ((uint32)dv<<1) ^ (uint32)(int32)(dv>>31);
		addvarint(&out, v);

		// pc delta
		addvarint(&out, (it.nextpc - it.pc) / it.pcscale);
	}
	
	// terminating value delta
	addvarint(&out, 0);

	free(d->p);
	*d = out;	
}


// pclntab initializes the pclntab symbol with
// runtime function and file name information.
void
pclntab(void)
{
	int32 i, nfunc, start, funcstart;
	LSym *ftab, *s;
	int32 off, end, frameptrsize;
	int64 funcdata_bytes;
	Pcln *pcln;
	Pciter it;
	static Pcln zpcln;
	
	funcdata_bytes = 0;
	ftab = linklookup(ctxt, "runtime.pclntab", 0);
	ftab->type = SPCLNTAB;
	ftab->reachable = 1;

	// See golang.org/s/go12symtab for the format. Briefly:
	//	8-byte header
	//	nfunc [PtrSize bytes]
	//	function table, alternating PC and offset to func struct [each entry PtrSize bytes]
	//	end PC [PtrSize bytes]
	//	offset to file table [4 bytes]
	nfunc = 0;
	for(ctxt->cursym = ctxt->textp; ctxt->cursym != nil; ctxt->cursym = ctxt->cursym->next)
		nfunc++;
	symgrow(ctxt, ftab, 8+PtrSize+nfunc*2*PtrSize+PtrSize+4);
	setuint32(ctxt, ftab, 0, 0xfffffffb);
	setuint8(ctxt, ftab, 6, MINLC);
	setuint8(ctxt, ftab, 7, PtrSize);
	setuintxx(ctxt, ftab, 8, nfunc, PtrSize);

	nfunc = 0;
	for(ctxt->cursym = ctxt->textp; ctxt->cursym != nil; ctxt->cursym = ctxt->cursym->next, nfunc++) {
		pcln = ctxt->cursym->pcln;
		if(pcln == nil)
			pcln = &zpcln;
	
		funcstart = ftab->np;
		funcstart += -ftab->np & (PtrSize-1);

		setaddr(ctxt, ftab, 8+PtrSize+nfunc*2*PtrSize, ctxt->cursym);
		setuintxx(ctxt, ftab, 8+PtrSize+nfunc*2*PtrSize+PtrSize, funcstart, PtrSize);

		// fixed size of struct, checked below
		off = funcstart;
		end = funcstart + PtrSize + 3*4 + 5*4 + pcln->npcdata*4 + pcln->nfuncdata*PtrSize;
		if(pcln->nfuncdata > 0 && (end&(PtrSize-1)))
			end += 4;
		symgrow(ctxt, ftab, end);

		// entry uintptr
		off = setaddr(ctxt, ftab, off, ctxt->cursym);

		// name int32
		off = setuint32(ctxt, ftab, off, ftabaddstring(ftab, ctxt->cursym->name));
		
		// args int32
		// TODO: Move into funcinfo.
		off = setuint32(ctxt, ftab, off, ctxt->cursym->args);
	
		// frame int32
		// TODO: Remove entirely. The pcsp table is more precise.
		// This is only used by a fallback case during stack walking
		// when a called function doesn't have argument information.
		// We need to make sure everything has argument information
		// and then remove this.
		frameptrsize = PtrSize;
		if(ctxt->cursym->leaf)
			frameptrsize = 0;
		off = setuint32(ctxt, ftab, off, ctxt->cursym->locals + frameptrsize);
		
		if(pcln != &zpcln) {
			renumberfiles(ctxt, pcln->file, pcln->nfile, &pcln->pcfile);
			if(0) {
				// Sanity check the new numbering
				for(pciterinit(ctxt, &it, &pcln->pcfile); !it.done; pciternext(&it)) {
					if(it.value < 1 || it.value > ctxt->nhistfile) {
						diag("bad file number in pcfile: %d not in range [1, %d]\n", it.value, ctxt->nhistfile);
						errorexit();
					}
				}
			}
		}

		// pcdata
		off = addpctab(ftab, off, &pcln->pcsp);
		off = addpctab(ftab, off, &pcln->pcfile);
		off = addpctab(ftab, off, &pcln->pcline);
		off = setuint32(ctxt, ftab, off, pcln->npcdata);
		off = setuint32(ctxt, ftab, off, pcln->nfuncdata);
		for(i=0; i<pcln->npcdata; i++)
			off = addpctab(ftab, off, &pcln->pcdata[i]);

		// funcdata, must be pointer-aligned and we're only int32-aligned.
		// Missing funcdata will be 0 (nil pointer).
		if(pcln->nfuncdata > 0) {
			if(off&(PtrSize-1))
				off += 4;
			for(i=0; i<pcln->nfuncdata; i++) {
				if(pcln->funcdata[i] == nil)
					setuintxx(ctxt, ftab, off+PtrSize*i, pcln->funcdataoff[i], PtrSize);
				else {
					// TODO: Dedup.
					funcdata_bytes += pcln->funcdata[i]->size;
					setaddrplus(ctxt, ftab, off+PtrSize*i, pcln->funcdata[i], pcln->funcdataoff[i]);
				}
			}
			off += pcln->nfuncdata*PtrSize;
		}

		if(off != end) {
			diag("bad math in functab: funcstart=%d off=%d but end=%d (npcdata=%d nfuncdata=%d ptrsize=%d)", funcstart, off, end, pcln->npcdata, pcln->nfuncdata, PtrSize);
			errorexit();
		}
	
		// Final entry of table is just end pc.
		if(ctxt->cursym->next == nil)
			setaddrplus(ctxt, ftab, 8+PtrSize+(nfunc+1)*2*PtrSize, ctxt->cursym, ctxt->cursym->size);
	}
	
	// Start file table.
	start = ftab->np;
	start += -ftab->np & (PtrSize-1);
	setuint32(ctxt, ftab, 8+PtrSize+nfunc*2*PtrSize+PtrSize, start);

	symgrow(ctxt, ftab, start+(ctxt->nhistfile+1)*4);
	setuint32(ctxt, ftab, start, ctxt->nhistfile);
	for(s = ctxt->filesyms; s != S; s = s->next)
		setuint32(ctxt, ftab, start + s->value*4, ftabaddstring(ftab, s->name));

	ftab->size = ftab->np;
	
	if(debug['v'])
		Bprint(&bso, "%5.2f pclntab=%lld bytes, funcdata total %lld bytes\n", cputime(), (vlong)ftab->size, (vlong)funcdata_bytes);
}	

enum {
	BUCKETSIZE = 256*MINFUNC,
	SUBBUCKETS = 16,
};

// findfunctab generates a lookup table to quickly find the containing
// function for a pc.  See src/runtime/symtab.go:findfunc for details.
void
findfunctab(void)
{
	LSym *t, *s;
	int32 idx, bidx, i, j, nbuckets;
	vlong min, max;

	t = linklookup(ctxt, "runtime.findfunctab", 0);
	t->type = SRODATA;
	t->reachable = 1;

	// find min and max address
	min = ctxt->textp->value;
	max = 0;
	for(s = ctxt->textp; s != nil; s = s->next)
		max = s->value + s->size;

	// allocate table
	nbuckets = (max-min+BUCKETSIZE-1)/BUCKETSIZE;
	symgrow(ctxt, t, nbuckets * (4+SUBBUCKETS));

	// fill in table
	s = ctxt->textp;
	idx = 0;
	for(i = 0; i < nbuckets; i++) {
		// Find first function which overlaps this bucket.
		// Only do leaf symbols; skip symbols which are just containers (sub != nil but outer == nil).
		while(s != nil && (s->value+s->size <= min + i * BUCKETSIZE || s->sub != nil && s->outer == nil)) {
			s = s->next;
			idx++;
		}
		// record this function in bucket header
		setuint32(ctxt, t, i*(4+SUBBUCKETS), idx);
		bidx = idx;

		// compute SUBBUCKETS deltas
		for(j = 0; j < SUBBUCKETS; j++) {
			while(s != nil && (s->value+s->size <= min + i * BUCKETSIZE + j * (BUCKETSIZE/SUBBUCKETS) || s->sub != nil && s->outer == nil)) {
				s = s->next;
				idx++;
			}
			if(idx - bidx >= 256)
				diag("too many functions in a findfunc bucket! %d %s", idx-bidx, s->name);
			setuint8(ctxt, t, i*(4+SUBBUCKETS)+4+j, idx-bidx);
		}
	}
}

// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	<u.h>
#include	<libc.h>
#include	<bio.h>
#include	<link.h>
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

static int
container(LSym *s)
{
	// We want to generate func table entries only for the "lowest level" symbols,
	// not containers of subsymbols.
	if(s != nil && s->sub != nil)
		return 1;
	return 0;
}

// pclntab initializes the pclntab symbol with
// runtime function and file name information.
void
pclntab(void)
{
	int32 i, nfunc, start, funcstart;
	LSym *ftab, *s, *last;
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
	//	nfunc [thearch.ptrsize bytes]
	//	function table, alternating PC and offset to func struct [each entry thearch.ptrsize bytes]
	//	end PC [thearch.ptrsize bytes]
	//	offset to file table [4 bytes]
	nfunc = 0;
	for(ctxt->cursym = ctxt->textp; ctxt->cursym != nil; ctxt->cursym = ctxt->cursym->next) {
		if(!container(ctxt->cursym))
			nfunc++;
	}
	symgrow(ctxt, ftab, 8+thearch.ptrsize+nfunc*2*thearch.ptrsize+thearch.ptrsize+4);
	setuint32(ctxt, ftab, 0, 0xfffffffb);
	setuint8(ctxt, ftab, 6, thearch.minlc);
	setuint8(ctxt, ftab, 7, thearch.ptrsize);
	setuintxx(ctxt, ftab, 8, nfunc, thearch.ptrsize);

	nfunc = 0;
	last = nil;
	for(ctxt->cursym = ctxt->textp; ctxt->cursym != nil; ctxt->cursym = ctxt->cursym->next) {
		last = ctxt->cursym;
		if(container(ctxt->cursym))
			continue;
		pcln = ctxt->cursym->pcln;
		if(pcln == nil)
			pcln = &zpcln;
	
		funcstart = ftab->np;
		funcstart += -ftab->np & (thearch.ptrsize-1);

		setaddr(ctxt, ftab, 8+thearch.ptrsize+nfunc*2*thearch.ptrsize, ctxt->cursym);
		setuintxx(ctxt, ftab, 8+thearch.ptrsize+nfunc*2*thearch.ptrsize+thearch.ptrsize, funcstart, thearch.ptrsize);

		// fixed size of struct, checked below
		off = funcstart;
		end = funcstart + thearch.ptrsize + 3*4 + 5*4 + pcln->npcdata*4 + pcln->nfuncdata*thearch.ptrsize;
		if(pcln->nfuncdata > 0 && (end&(thearch.ptrsize-1)))
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
		frameptrsize = thearch.ptrsize;
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
			if(off&(thearch.ptrsize-1))
				off += 4;
			for(i=0; i<pcln->nfuncdata; i++) {
				if(pcln->funcdata[i] == nil)
					setuintxx(ctxt, ftab, off+thearch.ptrsize*i, pcln->funcdataoff[i], thearch.ptrsize);
				else {
					// TODO: Dedup.
					funcdata_bytes += pcln->funcdata[i]->size;
					setaddrplus(ctxt, ftab, off+thearch.ptrsize*i, pcln->funcdata[i], pcln->funcdataoff[i]);
				}
			}
			off += pcln->nfuncdata*thearch.ptrsize;
		}

		if(off != end) {
			diag("bad math in functab: funcstart=%d off=%d but end=%d (npcdata=%d nfuncdata=%d ptrsize=%d)", funcstart, off, end, pcln->npcdata, pcln->nfuncdata, thearch.ptrsize);
			errorexit();
		}
	
		nfunc++;
	}
	// Final entry of table is just end pc.
	setaddrplus(ctxt, ftab, 8+thearch.ptrsize+nfunc*2*thearch.ptrsize, last, last->size);
	
	// Start file table.
	start = ftab->np;
	start += -ftab->np & (thearch.ptrsize-1);
	setuint32(ctxt, ftab, 8+thearch.ptrsize+nfunc*2*thearch.ptrsize+thearch.ptrsize, start);

	symgrow(ctxt, ftab, start+(ctxt->nhistfile+1)*4);
	setuint32(ctxt, ftab, start, ctxt->nhistfile);
	for(s = ctxt->filesyms; s != nil; s = s->next)
		setuint32(ctxt, ftab, start + s->value*4, ftabaddstring(ftab, s->name));

	ftab->size = ftab->np;
	
	if(debug['v'])
		Bprint(&bso, "%5.2f pclntab=%lld bytes, funcdata total %lld bytes\n", cputime(), (vlong)ftab->size, (vlong)funcdata_bytes);
}	

enum {
	BUCKETSIZE = 256*MINFUNC,
	SUBBUCKETS = 16,
	SUBBUCKETSIZE = BUCKETSIZE/SUBBUCKETS,
	NOIDX = 0x7fffffff
};

// findfunctab generates a lookup table to quickly find the containing
// function for a pc.  See src/runtime/symtab.go:findfunc for details.
void
findfunctab(void)
{
	LSym *t, *s, *e;
	int32 idx, i, j, nbuckets, n, base;
	vlong min, max, p, q;
	int32 *indexes;

	t = linklookup(ctxt, "runtime.findfunctab", 0);
	t->type = SRODATA;
	t->reachable = 1;

	// find min and max address
	min = ctxt->textp->value;
	max = 0;
	for(s = ctxt->textp; s != nil; s = s->next)
		max = s->value + s->size;

	// for each subbucket, compute the minimum of all symbol indexes
	// that map to that subbucket.
	n = (max-min+SUBBUCKETSIZE-1)/SUBBUCKETSIZE;
	indexes = (int32*)malloc(n*4);
	if(indexes == nil) {
		diag("out of memory");
		errorexit();
	}
	for(i = 0; i < n; i++)
		indexes[i] = NOIDX;
	idx = 0;
	for(s = ctxt->textp; s != nil; s = s->next) {
		if(container(s))
			continue;
		p = s->value;
		e = s->next;
		while(container(e))
			e = e->next;
		if(e != nil)
			q = e->value;
		else
			q = max;

		//print("%d: [%lld %lld] %s\n", idx, p, q, s->name);
		for(; p < q; p += SUBBUCKETSIZE) {
			i = (p - min) / SUBBUCKETSIZE;
			if(indexes[i] > idx)
				indexes[i] = idx;
		}
		i = (q - 1 - min) / SUBBUCKETSIZE;
		if(indexes[i] > idx)
			indexes[i] = idx;
		idx++;
	}

	// allocate table
	nbuckets = (max-min+BUCKETSIZE-1)/BUCKETSIZE;
	symgrow(ctxt, t, 4*nbuckets + n);

	// fill in table
	for(i = 0; i < nbuckets; i++) {
		base = indexes[i*SUBBUCKETS];
		if(base == NOIDX)
			diag("hole in findfunctab");
		setuint32(ctxt, t, i*(4+SUBBUCKETS), base);
		for(j = 0; j < SUBBUCKETS && i*SUBBUCKETS+j < n; j++) {
			idx = indexes[i*SUBBUCKETS+j];
			if(idx == NOIDX)
				diag("hole in findfunctab");
			if(idx - base >= 256) {
				diag("too many functions in a findfunc bucket! %d/%d %d %d", i, nbuckets, j, idx-base);
			}
			setuint8(ctxt, t, i*(4+SUBBUCKETS)+4+j, idx-base);
		}
	}
	free(indexes);
}

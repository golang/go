// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

// Runtime symbol table access.
// Very much a work in progress.

#define SYMCOUNTS ((int32*)(0x99LL<<32))	// known to 6l
#define SYMDATA ((byte*)(0x99LL<<32) + 8)

// Return a pointer to a byte array containing the symbol table segment.
//
// NOTE(rsc): I expect that we will clean up both the method of getting
// at the symbol table and the exact format of the symbol table at some
// point in the future.  It probably needs to be better integrated with
// the type strings table too.  This is just a quick way to get started
// and figure out what we want from/can do with it.
void
sys·symdat(Array *symtab, Array *pclntab)
{
	Array *a;
	int32 *v;

	v = SYMCOUNTS;

	a = mal(sizeof *a);
	a->nel = v[0];
	a->cap = a->nel;
	a->array = SYMDATA;
	symtab = a;
	FLUSH(&symtab);

	a = mal(sizeof *a);
	a->nel = v[1];
	a->cap = a->nel;
	a->array = SYMDATA + v[0];
	pclntab = a;
	FLUSH(&pclntab);
}

typedef struct Sym Sym;
struct Sym
{
	uint64 value;
	byte symtype;
	byte *name;
	byte *gotype;
};

// Walk over symtab, calling fn(&s) for each symbol.
void
walksymtab(void (*fn)(Sym*))
{
	int32 *v;
	byte *p, *ep, *q;
	Sym s;

	v = SYMCOUNTS;
	p = SYMDATA;
	ep = p + v[0];
	while(p < ep) {
		if(p + 7 > ep)
			break;
		s.value = ((uint32)p[0]<<24) | ((uint32)p[1]<<16) | ((uint32)p[2]<<8) | ((uint32)p[3]);
		if(!(p[4]&0x80))
			break;
		s.symtype = p[4] & ~0x80;
		p += 5;
		if(s.symtype == 'z' || s.symtype == 'Z') {
			// path reference string - skip first byte,
			// then 2-byte pairs ending at two zeros.
			// for now, just skip over it and ignore it.
			q = p+1;
			for(;;) {
				if(q+2 > ep)
					return;
				if(q[0] == '\0' && q[1] == '\0')
					break;
				q += 2;
			}
			p = q+2;
			s.name = nil;
		}else{
			q = mchr(p, '\0', ep);
			if(q == nil)
				break;
			s.name = p;
			p = q+1;
		}
		q = mchr(p, '\0', ep);
		if(q == nil)
			break;
		s.gotype = p;
		p = q+1;
		fn(&s);
	}
}

// Symtab walker; accumulates info about functions.

Func *func;
int32 nfunc;

static void
dofunc(Sym *sym)
{
	static byte *lastfuncname;
	static Func *lastfunc;
	Func *f;

	if(lastfunc && sym->symtype == 'm') {
		lastfunc->frame = sym->value;
		return;
	}
	if(sym->symtype != 'T' && sym->symtype != 't')
		return;
	if(strcmp(sym->name, (byte*)"etext") == 0)
		return;
	if(func == nil) {
		nfunc++;
		return;
	}

	f = &func[nfunc++];
	f->name = gostring(sym->name);
	f->entry = sym->value;
	lastfunc = f;
}

static void
buildfuncs(void)
{
	extern byte etext[];

	if(func != nil)
		return;
	nfunc = 0;
	walksymtab(dofunc);
	func = mal((nfunc+1)*sizeof func[0]);
	nfunc = 0;
	walksymtab(dofunc);
	func[nfunc].entry = (uint64)etext;
}

Func*
findfunc(uint64 addr)
{
	Func *f;
	int32 i, nf, n;

	if(func == nil)
		buildfuncs();
	if(nfunc == 0)
		return nil;
	if(addr < func[0].entry || addr >= func[nfunc].entry)
		return nil;

	// linear search, for debugging
	if(0) {
		for(i=0; i<nfunc; i++) {
			if(func[i].entry <= addr && addr < func[i+1].entry)
				return &func[i];
		}
		return nil;
	}

	// binary search to find func with entry <= addr.
	f = func;
	nf = nfunc;
	while(nf > 0) {
		n = nf/2;
		if(f[n].entry <= addr && addr < f[n+1].entry)
			return &f[n];
		else if(addr < f[n].entry)
			nf = n;
		else {
			f += n+1;
			nf -= n+1;
		}
	}

	// can't get here -- we already checked above
	// that the address was in the table bounds.
	// this can only happen if the table isn't sorted
	// by address or if the binary search above is buggy.
	prints("findfunc unreachable\n");
	return nil;
}

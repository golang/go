// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Runtime symbol table access.  Work in progress.
// The Plan 9 symbol table is not in a particularly convenient form.
// The routines here massage it into a more usable form; eventually
// we'll change 6l to do this for us, but it is easier to experiment
// here than to change 6l and all the other tools.
//
// The symbol table also needs to be better integrated with the type
// strings table in the future.  This is just a quick way to get started
// and figure out exactly what we want.

#include "runtime.h"

#define SYMCOUNTS ((int32*)(0x99LL<<32))	// known to 6l
#define SYMDATA ((byte*)(0x99LL<<32) + 8)

// Return a pointer to a byte array containing the symbol table segment.
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
static void
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
		s.name = p;
		if(s.symtype == 'z' || s.symtype == 'Z') {
			// path reference string - skip first byte,
			// then 2-byte pairs ending at two zeros.
			q = p+1;
			for(;;) {
				if(q+2 > ep)
					return;
				if(q[0] == '\0' && q[1] == '\0')
					break;
				q += 2;
			}
			p = q+2;
		}else{
			q = mchr(p, '\0', ep);
			if(q == nil)
				break;
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

static Func *func;
static int32 nfunc;

static byte **fname;
static int32 nfname;

static void
dofunc(Sym *sym)
{
	Func *f;

	switch(sym->symtype) {
	case 't':
	case 'T':
		if(strcmp(sym->name, (byte*)"etext") == 0)
			break;
		if(func == nil) {
			nfunc++;
			break;
		}
		f = &func[nfunc++];
		f->name = gostring(sym->name);
		f->entry = sym->value;
		break;
	case 'm':
		if(nfunc > 0 && func != nil)
			func[nfunc-1].frame = sym->value;
		break;
	case 'f':
		if(fname == nil) {
			if(sym->value >= nfname)
				nfname = sym->value+1;
			break;
		}
		fname[sym->value] = sym->name;
		break;
	}
}

// put together the path name for a z entry.
// the f entries have been accumulated into fname already.
static void
makepath(byte *buf, int32 nbuf, byte *path)
{
	int32 n, len;
	byte *p, *ep, *q;

	if(nbuf <= 0)
		return;

	p = buf;
	ep = buf + nbuf;
	*p = '\0';
	for(;;) {
		if(path[0] == 0 && path[1] == 0)
			break;
		n = (path[0]<<8) | path[1];
		path += 2;
		if(n >= nfname)
			break;
		q = fname[n];
		len = findnull(q);
		if(p+1+len >= ep)
			break;
		if(p > buf && p[-1] != '/')
			*p++ = '/';
		mcpy(p, q, len+1);
		p += len;
	}
}

// walk symtab accumulating path names for use by pc/ln table.
// don't need the full generality of the z entry history stack because
// there are no includes in go (and only sensible includes in our c).
static void
dosrcline(Sym *sym)
{
	static byte srcbuf[1000];
	static string srcstring;
	static int32 lno, incstart;
	static int32 nf, nhist;
	Func *f;

	switch(sym->symtype) {
	case 't':
	case 'T':
		f = &func[nf++];
		f->src = srcstring;
		f->ln0 += lno;
		break;
	case 'z':
		if(sym->value == 1) {
			// entry for main source file for a new object.
			makepath(srcbuf, sizeof srcbuf, sym->name+1);
			srcstring = gostring(srcbuf);
			lno = 0;
			nhist = 0;
		} else {
			// push or pop of included file.
			makepath(srcbuf, sizeof srcbuf, sym->name+1);
			if(srcbuf[0] != '\0') {
				if(nhist++ == 0)
					incstart = sym->value;
			}else{
				if(--nhist == 0)
					lno -= sym->value - incstart;
			}
		}
	}
}

enum { PcQuant = 1 };

// Interpret pc/ln table, saving the subpiece for each func.
static void
splitpcln(void)
{
	int32 line;
	uint64 pc;
	byte *p, *ep;
	Func *f, *ef;
	int32 *v;

	// pc/ln table bounds
	v = SYMCOUNTS;
	p = SYMDATA;
	p += v[0];
	ep = p+v[1];

	f = func;
	ef = func + nfunc;
	f->pcln.array = p;
	pc = func[0].entry;	// text base
	line = 0;
	for(; p < ep; p++) {
		if(f < ef && pc >= (f+1)->entry) {
			f->pcln.nel = p - f->pcln.array;
			f->pcln.cap = f->pcln.nel;
			f++;
			f->pcln.array = p;
			f->pc0 = pc;
			f->ln0 = line;
		}
		if(*p == 0) {
			// 4 byte add to line
			line += (p[1]<<24) | (p[2]<<16) | (p[3]<<8) | p[4];
			p += 4;
		} else if(*p <= 64) {
			line += *p;
		} else if(*p <= 128) {
			line -= *p - 64;
		} else {
			pc += PcQuant*(*p - 129);
		}
		pc += PcQuant;
	}
	if(f < ef) {
		f->pcln.nel = p - f->pcln.array;
		f->pcln.cap = f->pcln.nel;
	}
}


// Return actual file line number for targetpc in func f.
// (Source file is f->src.)
int32
funcline(Func *f, uint64 targetpc)
{
	byte *p, *ep;
	uint64 pc;
	int32 line;

	p = f->pcln.array;
	ep = p + f->pcln.nel;
	pc = f->pc0;
	line = f->ln0;
	for(; p < ep; p++) {
		if(pc >= targetpc)
			return line;
		if(*p == 0) {
			line += (p[1]<<24) | (p[2]<<16) | (p[3]<<8) | p[4];
			p += 4;
		} else if(*p <= 64) {
			line += *p;
		} else if(*p <= 128) {
			line -= *p - 64;
		} else {
			pc += PcQuant*(*p - 129);
		}
		pc += PcQuant;
	}
	return line;
}

static void
buildfuncs(void)
{
	extern byte etext[];

	if(func != nil)
		return;
	// count funcs, fnames
	nfunc = 0;
	nfname = 0;
	walksymtab(dofunc);

	// initialize tables
	func = mal((nfunc+1)*sizeof func[0]);
	func[nfunc].entry = (uint64)etext;
	fname = mal(nfname*sizeof fname[0]);
	nfunc = 0;
	walksymtab(dofunc);

	// split pc/ln table by func
	splitpcln();

	// record src file and line info for each func
	walksymtab(dosrcline);
}

Func*
findfunc(uint64 addr)
{
	Func *f;
	int32 nf, n;

	if(func == nil)
		buildfuncs();
	if(nfunc == 0)
		return nil;
	if(addr < func[0].entry || addr >= func[nfunc].entry)
		return nil;

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

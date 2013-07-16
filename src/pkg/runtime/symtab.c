// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Runtime symbol table parsing.
// See http://golang.org/s/go12symtab for an overview.

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"
#include "arch_GOARCH.h"
#include "malloc.h"

typedef struct Ftab Ftab;
struct Ftab
{
	uintptr	entry;
	Func	*func;
};

extern uintptr functab[];

static Ftab *ftab;
static uintptr nftab;
extern String *filetab[];
static uintptr nfiletab;

static String end = { (uint8*)"end", 3 };

void
runtime·symtabinit(void)
{
	int32 i, j;

	ftab = (Ftab*)(functab+1);
	nftab = functab[0];
	
	for(i=0; i<nftab; i++) {
		// NOTE: ftab[nftab].entry is legal; it is the address beyond the final function.
		if(ftab[i].entry > ftab[i+1].entry) {
			runtime·printf("function symbol table not sorted by program counter: %p %S > %p %S", ftab[i].entry, *ftab[i].func->name, ftab[i+1].entry, i+1 == nftab ? end : *ftab[i+1].func->name);
			for(j=0; j<=i; j++)
				runtime·printf("\t%p %S\n", ftab[j].entry, *ftab[j].func->name);
			runtime·throw("invalid runtime symbol table");
		}
	}
	nfiletab = (uintptr)filetab[0];
}

static uint32
readvarint(byte **pp)
{
	byte *p;
	uint32 v;
	int32 shift;
	
	v = 0;
	p = *pp;
	for(shift = 0;; shift += 7) {
		v |= (*p & 0x7F) << shift;
		if(!(*p++ & 0x80))
			break;
	}
	*pp = p;
	return v;
}

static uintptr
funcdata(Func *f, int32 i)
{
	byte *p;

	if(i < 0 || i >= f->nfuncdata)
		return 0;
	p = (byte*)&f->nfuncdata + 4 + f->npcdata*4;
	if(sizeof(void*) == 8 && ((uintptr)p & 4))
		p += 4;
	return ((uintptr*)p)[i];
}

// Return associated data value for targetpc in func f.
// (Source file is f->src.)
static int32
pcvalue(Func *f, int32 off, uintptr targetpc)
{
	byte *p;
	uintptr pc;
	int32 value, vdelta, pcshift;
	uint32 uvdelta, pcdelta;

	enum {
		debug = 0
	};

	switch(thechar) {
	case '5':
		pcshift = 2;
		break;
	default:	// 6, 8
		pcshift = 0;
		break;
	}

	// The table is a delta-encoded sequence of (value, pc) pairs.
	// Each pair states the given value is in effect up to pc.
	// The value deltas are signed, zig-zag encoded.
	// The pc deltas are unsigned.
	// The starting value is -1, the starting pc is the function entry.
	// The table ends at a value delta of 0 except in the first pair.
	if(off == 0)
		return -1;
	p = (byte*)f + off;
	pc = f->entry;
	value = -1;

	if(debug && !runtime·panicking)
		runtime·printf("pcvalue start f=%S [%p] pc=%p targetpc=%p value=%d tab=%p\n",
			*f->name, f, pc, targetpc, value, p);
	
	for(;;) {
		uvdelta = readvarint(&p);
		if(uvdelta == 0 && pc != f->entry)
			break;
		if(uvdelta&1)
			uvdelta = ~(uvdelta>>1);
		else
			uvdelta >>= 1;
		vdelta = (int32)uvdelta;
		pcdelta = readvarint(&p) << pcshift;
		value += vdelta;
		pc += pcdelta;
		if(debug)
			runtime·printf("\tvalue=%d until pc=%p\n", value, pc);
		if(targetpc < pc)
			return value;
	}
	
	// If there was a table, it should have covered all program counters.
	// If not, something is wrong.
	runtime·printf("runtime: invalid pc-encoded table f=%S pc=%p targetpc=%p tab=%p\n",
		*f->name, pc, targetpc, p);
	runtime·throw("invalid runtime symbol table");
	return -1;
}

static String unknown = { (uint8*)"?", 1 };

int32
runtime·funcline(Func *f, uintptr targetpc, String *file)
{
	int32 line;
	int32 fileno;

	*file = unknown;
	fileno = pcvalue(f, f->pcfile, targetpc);
	line = pcvalue(f, f->pcln, targetpc);
	if(fileno == -1 || line == -1 || fileno >= nfiletab) {
		// runtime·printf("looking for %p in %S got file=%d line=%d\n", targetpc, *f->name, fileno, line);
		return 0;
	}
	*file = *filetab[fileno];
	return line;
}

int32
runtime·funcspdelta(Func *f, uintptr targetpc)
{
	int32 x;
	
	x = pcvalue(f, f->pcsp, targetpc);
	if(x&(sizeof(void*)-1))
		runtime·printf("invalid spdelta %d %d\n", f->pcsp, x);
	return x;
}

static int32
pcdatavalue(Func *f, int32 table, uintptr targetpc)
{
	if(table < 0 || table >= f->npcdata)
		return -1;
	return pcvalue(f, (&f->nfuncdata)[1+table], targetpc);
}

int32
runtime·funcarglen(Func *f, uintptr targetpc)
{
	return pcdatavalue(f, 0, targetpc);
}

void
runtime·funcline_go(Func *f, uintptr targetpc, String retfile, intgo retline)
{
	retline = runtime·funcline(f, targetpc, &retfile);
	FLUSH(&retline);
}

void
runtime·funcname_go(Func *f, String ret)
{
	ret = *f->name;
	FLUSH(&ret);
}

void
runtime·funcentry_go(Func *f, uintptr ret)
{
	ret = f->entry;
	FLUSH(&ret);
}

Func*
runtime·findfunc(uintptr addr)
{
	Ftab *f;
	int32 nf, n;

	if(nftab == 0)
		return nil;
	if(addr < ftab[0].entry || addr >= ftab[nftab].entry)
		return nil;

	// binary search to find func with entry <= addr.
	f = ftab;
	nf = nftab;
	while(nf > 0) {
		n = nf/2;
		if(f[n].entry <= addr && addr < f[n+1].entry)
			return f[n].func;
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
	runtime·prints("findfunc unreachable\n");
	return nil;
}

static bool
hasprefix(String s, int8 *p)
{
	int32 i;

	for(i=0; i<s.len; i++) {
		if(p[i] == 0)
			return 1;
		if(p[i] != s.str[i])
			return 0;
	}
	return p[i] == 0;
}

static bool
contains(String s, int8 *p)
{
	int32 i;

	if(p[0] == 0)
		return 1;
	for(i=0; i<s.len; i++) {
		if(s.str[i] != p[0])
			continue;
		if(hasprefix((String){s.str + i, s.len - i}, p))
			return 1;
	}
	return 0;
}

bool
runtime·showframe(Func *f, G *gp)
{
	static int32 traceback = -1;

	if(m->throwing && gp != nil && (gp == m->curg || gp == m->caughtsig))
		return 1;
	if(traceback < 0)
		traceback = runtime·gotraceback(nil);
	return traceback > 1 || f != nil && contains(*f->name, ".") && !hasprefix(*f->name, "runtime.");
}

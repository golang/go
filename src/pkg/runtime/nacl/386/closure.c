// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * Closure implementation for Native Client.
 * Native Client imposes some interesting restrictions.
 *
 * First, we can only add new code to the code segment
 * through a special system call, and we have to pick the
 * maximum amount of code we're going to add that way
 * at link time (8l reserves 512 kB for us).
 *
 * Second, once we've added the code we can't ever
 * change it or delete it.  If we want to garbage collect
 * the memory and then reuse it for another closure,
 * we have to do so without editing the code.
 *
 * To address both of these, we fill the code segment pieces
 * with very stylized closures.  Each has the form given below
 * in the comments on the closasm array, with ** replaced by
 * a pointer to a single word of memory.  The garbage collector
 * treats a pointer to such a closure as equivalent to the value
 * held in **.  This tiled run of closures is called the closure array.
 *
 * The ptr points at a ClosureData structure, defined below,
 * which gives the function, arguments, and size for the
 * closuretramp function.  The ClosureData structure has
 * in it a pointer to a ClosureFreeList structure holding the index
 * of the closure in the closure array (but not a pointer to it). 
 * That structure has a finalizer: when the garbage collector
 * notices that the ClosureFreeList structure is not referenced
 * anymore, that means the closure is not referenced, so it
 * can be reused.  To do that, the ClosureFreeList entry is put
 * onto an actual free list.
 */
#include "runtime.h"
#include "malloc.h"

// NaCl system call to copy data into text segment.
extern int32 dyncode_copy(void*, void*, int32);

enum{
	// Allocate chunks of 4096 bytes worth of closures:
	// at 64 bytes each, that's 64 closures.
	ClosureChunk = 4096,
	ClosureSize = 64,
};

typedef struct ClosureFreeList ClosureFreeList;
struct ClosureFreeList
{
	ClosureFreeList *next;
	int32 index;	// into closure array
};

// Known to closasm
typedef struct ClosureData ClosureData;
struct ClosureData
{
	ClosureFreeList *free;
	byte *fn;
	int32 siz;
	// then args
};

// List of the closure data pointer blocks we've allocated
// and hard-coded in the closure text segments.
// The list keeps the pointer blocks from getting collected.
typedef struct ClosureDataList ClosureDataList;
struct ClosureDataList
{
	ClosureData **block;
	ClosureDataList *next;
};

static struct {
	Lock;
	byte *code;
	byte *ecode;
	ClosureFreeList *free;
	ClosureDataList *datalist;
	byte buf[ClosureChunk];
} clos;

static byte closasm[64] = {
	0x8b, 0x1d, 0, 0, 0, 0,	// MOVL **, BX
	0x8b, 0x4b, 8,		// MOVL 8(BX), CX
	0x8d, 0x73, 12,		// LEAL 12(BX), SI
	0x29, 0xcc,		// SUBL CX, SP
	0x89, 0xe7,		// MOVL SP, DI
	0xc1, 0xe9, 2,		// SHRL $2, CX
	0xf3, 0xa5,		// REP MOVSL
	0x8b, 0x5b, 4,		// MOVL 4(BX), BX
	0x90, 0x90, 0x90,	// NOP...
	0x83, 0xe3, ~31,	// ANDL $~31, BX
	0xff, 0xd3,		// CALL *BX
	// --- 32-byte boundary
	0x8b, 0x1d, 0, 0, 0, 0,	// MOVL **, BX
	0x03, 0x63, 8,		// ADDL 8(BX), SP
	0x5b,			// POPL BX
	0x83, 0xe3, ~31,	// ANDL $~31, BX
	0xff, 0xe3,		// JMP *BX
	0xf4,			// HLT...
	0xf4, 0xf4, 0xf4, 0xf4,
	0xf4, 0xf4, 0xf4, 0xf4,
	0xf4, 0xf4, 0xf4, 0xf4,
	0xf4, 0xf4, 0xf4, 0xf4,
	// --- 32-byte boundary
};

// Returns immediate pointer from closure code block.
// Triple pointer:
//	p is the instruction stream
//	p+2 is the location of the immediate value
//	*(p+2) is the immediate value, a word in the pointer block
//		permanently associated with this closure.
//	**(p+2) is the ClosureData* pointer temporarily associated
//		with this closure.
//
#define codeptr(p) *(ClosureData***)((byte*)(p)+2)

void
runtime·finclosure(void *v)
{
	byte *p;
	ClosureFreeList *f;

	f = v;
	p = clos.code + f->index*ClosureSize;
	*codeptr(p) = nil;

	runtime·lock(&clos);
	f->next = clos.free;
	clos.free = f;
	runtime·unlock(&clos);
}

#pragma textflag 7
// func closure(siz int32,
//	fn func(arg0, arg1, arg2 *ptr, callerpc uintptr, xxx) yyy,
//	arg0, arg1, arg2 *ptr) (func(xxx) yyy)
void
runtime·closure(int32 siz, byte *fn, byte *arg0)
{
	byte *p, **ret;
	int32 e, i, n, off;
	extern byte rodata[], etext[];
	ClosureData *d, **block;
	ClosureDataList *l;
	ClosureFreeList *f;

	if(siz < 0 || siz%4 != 0)
		runtime·throw("bad closure size");

	ret = (byte**)((byte*)&arg0 + siz);

	if(siz > 100) {
		// TODO(rsc): implement stack growth preamble?
		runtime·throw("closure too big");
	}

	runtime·lock(&clos);
	if(clos.free == nil) {
		// Allocate more closures.
		if(clos.code == nil) {
			// First time: find closure space, between end of text
			// segment and beginning of data.
			clos.code = (byte*)(((uintptr)etext + 65535) & ~65535);
			clos.ecode = clos.code;
			mheap.closure_min = clos.code;
			mheap.closure_max = rodata;
		}
		if(clos.ecode+ClosureChunk > rodata) {
			// Last ditch effort: garbage collect and hope.
			runtime·unlock(&clos);
			runtime·gc(1);
			runtime·lock(&clos);
			if(clos.free != nil)
				goto alloc;
			runtime·throw("ran out of room for closures in text segment");
		}

		n = ClosureChunk/ClosureSize;
		
		// Allocate the pointer block as opaque to the
		// garbage collector.  Finalizers will clean up.
		block = runtime·mallocgc(n*sizeof block[0], RefNoPointers, 1, 1);

		// Pointers into the pointer block are getting added
		// to the text segment; keep a pointer here in the data
		// segment so that the garbage collector doesn't free
		// the block itself.
		l = runtime·mal(sizeof *l);
		l->block = block;
		l->next = clos.datalist;
		clos.datalist = l;

		p = clos.buf;
		off = (clos.ecode - clos.code)/ClosureSize;
		for(i=0; i<n; i++) {
			f = runtime·mal(sizeof *f);
			f->index = off++;
			f->next = clos.free;
			clos.free = f;

			// There are two hard-coded immediate values in
			// the assembly that need to be pp+i, one 2 bytes in
			// and one 2 bytes after the 32-byte boundary.
			runtime·mcpy(p, closasm, ClosureSize);
			*(ClosureData***)(p+2) = block+i;
			*(ClosureData***)(p+32+2) = block+i;
			p += ClosureSize;
		}

		if(p != clos.buf+sizeof clos.buf)
			runtime·throw("bad buf math in closure");

		e = runtime·dyncode_copy(clos.ecode, clos.buf, ClosureChunk);
		if(e != 0) {
			fd = 2;
			runtime·printf("dyncode_copy: error %d\n", e);
			runtime·throw("dyncode_copy");
		}
		clos.ecode += ClosureChunk;
	}

alloc:
	// Grab a free closure and save the data pointer in its indirect pointer.
	f = clos.free;
	clos.free = f->next;
	f->next = nil;
	p = clos.code + f->index*ClosureSize;

	d = runtime·mal(sizeof(*d)+siz);
	d->free = f;
	d->fn = fn;
	d->siz = siz;
	runtime·mcpy((byte*)(d+1), (byte*)&arg0, siz);
	*codeptr(p) = d;
	runtime·addfinalizer(f, finclosure, 0);
	runtime·unlock(&clos);

	*ret = p;
}



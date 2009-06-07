// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs.h"

// Stubs for memory management.
// In a separate file so they can be overridden during testing of gc.

enum
{
	NHUNK		= 20<<20,
};

// Convenient wrapper around mmap.
static void*
brk(uint32 n)
{
	byte *v;

	v = sys_mmap(nil, n, PROT_READ|PROT_WRITE|PROT_EXEC, MAP_ANON|MAP_PRIVATE, 0, 0);
	m->mem.nmmap += n;
	return v;
}

// Allocate n bytes of memory.  Note that this gets used
// to allocate new stack segments, so at each call to a function
// you have to ask yourself "would it be okay to call mal recursively
// right here?"  The answer is yes unless we're in the middle of
// editing the malloc state in m->mem.
void*
oldmal(uint32 n)
{
	byte* v;

	// round to keep everything 64-bit aligned
	n = rnd(n, 8);

	// be careful.  calling any function might invoke
	// mal to allocate more stack.
	if(n > NHUNK) {
		v = brk(n);
	} else {
		// allocate a new hunk if this one is too small
		if(n > m->mem.nhunk) {
			// here we're in the middle of editing m->mem
			// (we're about to overwrite m->mem.hunk),
			// so we can't call brk - it might call mal to grow the
			// stack, and the recursive call would allocate a new
			// hunk, and then once brk returned we'd immediately
			// overwrite that hunk with our own.
			// (the net result would be a memory leak, not a crash.)
			// so we have to call sys_mmap directly - it is written
			// in assembly and tagged not to grow the stack.
			m->mem.hunk =
				sys_mmap(nil, NHUNK, PROT_READ|PROT_WRITE|PROT_EXEC,
					MAP_ANON|MAP_PRIVATE, 0, 0);
			m->mem.nhunk = NHUNK;
			m->mem.nmmap += NHUNK;
		}
		v = m->mem.hunk;
		m->mem.hunk += n;
		m->mem.nhunk -= n;
	}
	m->mem.nmal += n;
	return v;
}

void
sys_mal(uint32 n, uint8 *ret)
{
	ret = mal(n);
	FLUSH(&ret);
}

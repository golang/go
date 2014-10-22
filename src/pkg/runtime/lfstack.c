// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Lock-free stack.
// The following code runs only on g0 stack.

#include "runtime.h"
#include "arch_GOARCH.h"

#ifdef _64BIT
// Amd64 uses 48-bit virtual addresses, 47-th bit is used as kernel/user flag.
// So we use 17msb of pointers as ABA counter.
# define PTR_BITS 47
#else
# define PTR_BITS 32
#endif
#define PTR_MASK ((1ull<<PTR_BITS)-1)
#define CNT_MASK (0ull-1)

#ifdef _64BIT
#ifdef GOOS_solaris
// SPARC64 and Solaris on AMD64 uses all 64 bits of virtual addresses.
// Use low-order three bits as ABA counter.
// http://docs.oracle.com/cd/E19120-01/open.solaris/816-5138/6mba6ua5p/index.html
#undef PTR_BITS
#undef CNT_MASK
#undef PTR_MASK
#define PTR_BITS 0
#define CNT_MASK 7
#define PTR_MASK ((0ull-1)<<3)
#endif
#endif

void
runtime·lfstackpush(uint64 *head, LFNode *node)
{
	uint64 old, new;

	if((uintptr)node != ((uintptr)node&PTR_MASK)) {
		runtime·printf("p=%p\n", node);
		runtime·throw("runtime·lfstackpush: invalid pointer");
	}

	node->pushcnt++;
	new = (uint64)(uintptr)node|(((uint64)node->pushcnt&CNT_MASK)<<PTR_BITS);
	for(;;) {
		old = runtime·atomicload64(head);
		node->next = (LFNode*)(uintptr)(old&PTR_MASK);
		if(runtime·cas64(head, old, new))
			break;
	}
}

LFNode*
runtime·lfstackpop(uint64 *head)
{
	LFNode *node, *node2;
	uint64 old, new;

	for(;;) {
		old = runtime·atomicload64(head);
		if(old == 0)
			return nil;
		node = (LFNode*)(uintptr)(old&PTR_MASK);
		node2 = runtime·atomicloadp(&node->next);
		new = 0;
		if(node2 != nil)
			new = (uint64)(uintptr)node2|(((uint64)node2->pushcnt&CNT_MASK)<<PTR_BITS);
		if(runtime·cas64(head, old, new))
			return node;
	}
}

void
runtime·lfstackpush_m(void)
{
	runtime·lfstackpush(g->m->ptrarg[0], g->m->ptrarg[1]);
	g->m->ptrarg[0] = nil;
	g->m->ptrarg[1] = nil;
}

void
runtime·lfstackpop_m(void)
{
	g->m->ptrarg[0] = runtime·lfstackpop(g->m->ptrarg[0]);
}

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Lock-free stack.

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
	old = runtime·atomicload64(head);
	for(;;) {
		node->next = (LFNode*)(uintptr)(old&PTR_MASK);
		if(runtime·cas64(head, &old, new))
			break;
	}
}

LFNode*
runtime·lfstackpop(uint64 *head)
{
	LFNode *node, *node2;
	uint64 old, new;

	old = runtime·atomicload64(head);
	for(;;) {
		if(old == 0)
			return nil;
		node = (LFNode*)(uintptr)(old&PTR_MASK);
		node2 = runtime·atomicloadp(&node->next);
		new = 0;
		if(node2 != nil)
			new = (uint64)(uintptr)node2|(((uint64)node2->pushcnt&CNT_MASK)<<PTR_BITS);
		if(runtime·cas64(head, &old, new))
			return node;
	}
}

void
runtime·lfstackpop2(uint64 *head, LFNode *node)
{
	node = runtime·lfstackpop(head);
	FLUSH(&node);
}

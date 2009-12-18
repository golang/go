// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

// returns number of bytes that the fp instruction is occupying
static uint32
isfltinstr(uint32 *pc)
{
	uint32 i;
	uint32 c;
	
	i = *pc;
	c = i >> 25 & 7;
	
	switch(c) {
	case 6: // 110
//printf(" %p coproc multi: %x\n", pc, i);
		return 4;
	case 7: // 111
		if (i>>24 & 1) return 0; // ignore swi
//printf(" %p coproc %x\n", pc, i);
		return 4;
	}

	// lookahead for virtual instructions that span multiple arm instructions
	c = ((*pc & 0x0f000000) >> 16) |
		((*(pc + 1)  & 0x0f000000) >> 20) |
		((*(pc + 2) & 0x0f000000) >> 24);
	if(c == 0x50d) {
//printf(" %p coproc const %x\n", pc, i);
		return 12;
	}

//printf(" %p %x\n", pc, i);
	return 0;
}

#pragma textflag 7
uint32*
_sfloat2(uint32 *lr, uint32 r0)
{
	uint32 skip;
	
//printf("softfloat: pre %p\n", lr);
	while(skip = isfltinstr(lr))
		lr += skip;
//printf(" post: %p\n", lr);
	return lr;
}



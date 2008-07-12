// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

extern int32	debug;

static int8 spmark[] = "\xa7\xf1\xd9\x2a\x82\xc8\xd8\xfe";

extern void morestack2();

void
traceback(uint8 *pc, uint8 *sp, void* r15)
{
	int32 spoff;
	int8* spp;
	uint8* callpc;
	int32 counter;
	int32 i;
	int8* name;
	G g;
	Stktop *stktop;

	// store local copy of per-process data block that we can write as we unwind
	mcpy((byte*)&g, (byte*)r15, sizeof(G));

	counter = 0;
	name = "panic";
	for(;;){
		callpc = pc;
		if((uint8*)morestack2 == pc) {
			// call site in _morestack(); pop to earlier stack block to get true caller
			stktop = (Stktop*)g.stackbase;
			g.stackbase = stktop->oldbase;
			g.stackguard = stktop->oldguard;
			sp = stktop->oldsp;
			pc = ((uint8**)sp)[1];
			sp += 16;  // two irrelevant calls on stack - morestack, plus the call morestack made
			continue;
		}
		/* find SP offset by stepping back through instructions to SP offset marker */
		while(pc > (uint8*)0x1000+sizeof spmark-1) {
			for(spp = spmark; *spp != '\0' && *pc++ == (uint8)*spp++; )
				;
			if(*spp == '\0'){
				spoff = *pc++;
				spoff += *pc++ << 8;
				spoff += *pc++ << 16;
				name = (int8*)pc;
				sp += spoff + 8;
				break;
			}
		}
		if(counter++ > 100){
			prints("stack trace terminated\n");
			break;
		}
		if((pc = ((uint8**)sp)[-1]) <= (uint8*)0x1000)
			break;

		/* print this frame */
		prints("0x");
		sys·printpointer(callpc);
		prints("?zi\n");
		prints("\t");
		prints(name);
		prints("(");
		for(i = 0; i < 3; i++){
			if(i != 0)
				prints(", ");
			sys·printint(((uint32*)sp)[i]);
		}
		prints(", ...)\n");
		prints("\t");
		prints(name);
		prints("(");
		for(i = 0; i < 3; i++){
			if(i != 0)
				prints(", ");
			prints("0x");
			sys·printpointer(((void**)sp)[i]);
		}
		prints(", ...)\n");
	}
}

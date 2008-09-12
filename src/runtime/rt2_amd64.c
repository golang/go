// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

extern int32	debug;

static int8 spmark[] = "\xa7\xf1\xd9\x2a\x82\xc8\xd8\xfe";

extern uint8 end;

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
		if((uint8*)retfromnewstack == pc) {
			// call site is retfromnewstack(); pop to earlier stack block to get true caller
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
			if(pc >= &end)
				return;
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

/*
 * For trace traps, disassemble instruction to see if it's INTB of known type.
 */
int32
inlinetrap(int32 sig, byte* pc)
{
	extern void etext();
	extern void _rt0_amd64_darwin();

	if(sig != 5)	/* SIGTRAP */
		return 0;
	if(pc-2 < (byte*)_rt0_amd64_darwin || pc >= (byte*)etext)
		return 0;
	if(pc[-2] != 0xcd)  /* INTB */
		return 0;
	switch(pc[-1]) {
	case 5:
		prints("\nTRAP: array out of bounds\n");
		break;
	case 6:
		prints("\nTRAP: leaving function with returning a value\n");
		break;
	default:
		prints("\nTRAP: unknown run-time trap ");
		sys·printint(pc[-1]);
		prints("\n");
	}
	return 1;
}

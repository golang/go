// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

extern int32	debug;

extern uint8 end;

void
traceback(uint8 *pc, uint8 *sp, void* r15)
{
	uint8* callpc;
	int32 counter;
	int32 i;
	string name;
	Func *f;
	G g;
	Stktop *stktop;

	// store local copy of per-process data block that we can write as we unwind
	mcpy((byte*)&g, (byte*)r15, sizeof(G));

	// if the PC is zero, it's probably due to a nil function pointer.
	// pop the failed frame.
	if(pc == nil) {
		pc = ((uint8**)sp)[0];
		sp += 8;
	}

	counter = 0;
	name = gostring((byte*)"panic");
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
		f = findfunc((uint64)callpc);
		if(f == nil)
			return;
		name = f->name;
		sp += f->frame;
		if(counter++ > 100){
			prints("stack trace terminated\n");
			break;
		}
		if((pc = ((uint8**)sp)[-1]) <= (uint8*)0x1000)
			break;

		/* print this frame */
		prints("0x");
		sys·printpointer(callpc  - 1);	// -1 to get to CALL instr.
		prints("?zi ");
		sys·printstring(f->src);
		prints(":");
		sys·printint(funcline(f, (uint64)callpc-1));	// -1 to get to CALL instr.
		prints("\n");
		prints("\t");
		sys·printstring(name);
		prints("(");
		for(i = 0; i < 3; i++){
			if(i != 0)
				prints(", ");
			sys·printint(((uint32*)sp)[i]);
		}
		prints(", ...)\n");
		prints("\t");
		sys·printstring(name);
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

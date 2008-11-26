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
		if(f == nil) {
			printf("%p unknown pc\n", callpc);
			return;
		}
		name = f->name;
		if(f->frame < 8)	// assembly funcs say 0 but lie
			sp += 8;
		else
			sp += f->frame;
		if(counter++ > 100){
			prints("stack trace terminated\n");
			break;
		}
		if((pc = ((uint8**)sp)[-1]) <= (uint8*)0x1000)
			break;

		// print this frame
		//	main+0xf /home/rsc/go/src/runtime/x.go:23
		//		main(0x1, 0x2, 0x3)
		printf("%S", name);
		if((uint64)callpc > f->entry)
			printf("+%X", (uint64)callpc - f->entry);
		printf(" %S:%d\n", f->src, funcline(f, (uint64)callpc-1));	// -1 to get to CALL instr.
		printf("\t%S(", name);
		for(i = 0; i < f->args; i++) {
			if(i != 0)
				prints(", ");
			sys·printhex(((uint32*)sp)[i]);
			if(i >= 4) {
				prints(", ...");
				break;
			}
		}
		prints(")\n");
	}
}

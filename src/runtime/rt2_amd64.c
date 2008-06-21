// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

extern int32	debug;

static int8 spmark[] = "\xa7\xf1\xd9\x2a\x82\xc8\xd8\xfe";

void
traceback(uint8 *pc, uint8 *sp)
{
	int32 spoff;
	int8* spp;
	int32 counter;
	int32 i;
	int8* name;


	counter = 0;
	name = "panic";
	for(;;){
		prints("0x");
		sys_printpointer(pc);
		prints("?zi\n");
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
		/* print args for this frame */
		prints("\t");
		prints(name);
		prints("(");
		for(i = 0; i < 3; i++){
			if(i != 0)
				prints(", ");
			sys_printint(((uint32*)sp)[i]);
		}
		prints(", ...)\n");
		prints("\t");
		prints(name);
		prints("(");
		for(i = 0; i < 3; i++){
			if(i != 0)
				prints(", ");
			prints("0x");
			sys_printpointer(((void**)sp)[i]);
		}
		prints(", ...)\n");
		/* print pc for next frame */
	}
}

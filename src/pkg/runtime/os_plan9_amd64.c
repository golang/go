// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file. 

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"
#include "signals_GOOS.h"

void
runtime·dumpregs(Ureg *u)
{
	runtime·printf("ax	%X\n", u->ax);
	runtime·printf("bx	%X\n", u->bx);
	runtime·printf("cx	%X\n", u->cx);
	runtime·printf("dx	%X\n", u->dx);
	runtime·printf("di	%X\n", u->di);
	runtime·printf("si	%X\n", u->si);
	runtime·printf("bp	%X\n", u->bp);
	runtime·printf("sp	%X\n", u->sp);
	runtime·printf("r8	%X\n", u->r8);
	runtime·printf("r9	%X\n", u->r9);
	runtime·printf("r10	%X\n", u->r10);
	runtime·printf("r11	%X\n", u->r11);
	runtime·printf("r12	%X\n", u->r12);
	runtime·printf("r13	%X\n", u->r13);
	runtime·printf("r14	%X\n", u->r14);
	runtime·printf("r15	%X\n", u->r15);
	runtime·printf("ip	%X\n", u->ip);
	runtime·printf("flags	%X\n", u->flags);
	runtime·printf("cs	%X\n", (uint64)u->cs);
	runtime·printf("fs	%X\n", (uint64)u->fs);
	runtime·printf("gs	%X\n", (uint64)u->gs);
}

int32
runtime·sighandler(void *v, int8 *s, G *gp)
{
	bool crash;
	Ureg *ureg;
	uintptr *sp;
	SigTab *sig, *nsig;
	int32 len, i;

	if(!s)
		return NCONT;
			
	len = runtime·findnull((byte*)s);
	if(len <= 4 || runtime·mcmp((byte*)s, (byte*)"sys:", 4) != 0)
		return NDFLT;

	nsig = nil;
	sig = runtime·sigtab;
	for(i=0; i < NSIG; i++) {
		if(runtime·strstr((byte*)s, (byte*)sig->name)) {
			nsig = sig;
			break;
		}
		sig++;
	}

	if(nsig == nil)
		return NDFLT;

	ureg = v;
	if(nsig->flags & SigPanic) {
		if(gp == nil || m->notesig == 0)
			goto Throw;

		// Save error string from sigtramp's stack,
		// into gsignal->sigcode0, so we can reliably
		// access it from the panic routines.
		if(len > ERRMAX)
			len = ERRMAX;
		runtime·memmove((void*)m->notesig, (void*)s, len);

		gp->sig = i;
		gp->sigpc = ureg->ip;

		// Only push runtime·sigpanic if ureg->ip != 0.
		// If ureg->ip == 0, probably panicked because of a
		// call to a nil func.  Not pushing that onto sp will
		// make the trace look like a call to runtime·sigpanic instead.
		// (Otherwise the trace will end at runtime·sigpanic and we
		// won't get to see who faulted.)
		if(ureg->ip != 0) {
			sp = (uintptr*)ureg->sp;
			*--sp = ureg->ip;
			ureg->sp = (uint64)sp;
		}
		ureg->ip = (uintptr)runtime·sigpanic;
		return NCONT;
	}

	if(!(nsig->flags & SigThrow))
		return NDFLT;

Throw:
	runtime·startpanic();

	runtime·printf("%s\n", s);
	runtime·printf("PC=%X\n", ureg->ip);
	runtime·printf("\n");

	if(runtime·gotraceback(&crash)) {
		runtime·traceback((void*)ureg->ip, (void*)ureg->sp, 0, gp);
		runtime·tracebackothers(gp);
		runtime·dumpregs(ureg);
	}
	
	if(crash)
		runtime·crash();

	runtime·goexitsall("");
	runtime·exits(s);

	return 0;
}

void
runtime·sigenable(uint32 sig)
{
	USED(sig);
}

void
runtime·sigdisable(uint32 sig)
{
	USED(sig);
}

void
runtime·resetcpuprofiler(int32 hz)
{
	// TODO: Enable profiling interrupts.
	
	m->profilehz = hz;
}

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
runtime·sighandler(void *v, int8 *note, G *gp)
{
	uintptr *sp;
	SigTab *t;
	bool crash;
	Ureg *ureg;
	intgo len, n;
	int32 sig, flags;

	ureg = (Ureg*)v;

	// The kernel will never pass us a nil note or ureg so we probably
	// made a mistake somewhere in runtime·sigtramp.
	if(ureg == nil || note == nil) {
		runtime·printf("sighandler: ureg %p note %p\n", ureg, note);
		goto Throw;
	}

	// Check that the note is no more than ERRMAX bytes (including
	// the trailing NUL). We should never receive a longer note.
	len = runtime·findnull((byte*)note);
	if(len > ERRMAX-1) {
		runtime·printf("sighandler: note is longer than ERRMAX\n");
		goto Throw;
	}

	// See if the note matches one of the patterns in runtime·sigtab.
	// Notes that do not match any pattern can be handled at a higher
	// level by the program but will otherwise be ignored.
	flags = SigNotify;
	for(sig = 0; sig < nelem(runtime·sigtab); sig++) {
		t = &runtime·sigtab[sig];
		n = runtime·findnull((byte*)t->name);
		if(len < n)
			continue;
		if(runtime·strncmp((byte*)note, (byte*)t->name, n) == 0) {
			flags = t->flags;
			break;
		}
	}

	if(flags & SigGoExit)
		runtime·exits(note+9); // Strip "go: exit " prefix.

	if(flags & SigPanic) {
		// Copy the error string from sigtramp's stack into m->notesig so
		// we can reliably access it from the panic routines.
		runtime·memmove(g->m->notesig, note, len+1);

		gp->sig = sig;
		gp->sigpc = ureg->ip;

		// Only push runtime·sigpanic if PC != 0.
		//
		// If PC == 0, probably panicked because of a call to a nil func.
		// Not pushing that onto SP will make the trace look like a call
		// to runtime·sigpanic instead. (Otherwise the trace will end at
		// runtime·sigpanic and we won't get to see who faulted).
		if(ureg->ip != 0) {
			sp = (uintptr*)ureg->sp;
			*--sp = ureg->ip;
			ureg->sp = (uint64)sp;
		}
		ureg->ip = (uintptr)runtime·sigpanic;
		return NCONT;
	}

	if(flags & SigNotify) {
		// TODO(ality): See if os/signal wants it.
		//if(runtime·sigsend(...))
		//	return NCONT;
	}
	if(flags & SigKill)
		goto Exit;
	if(!(flags & SigThrow))
		return NCONT;

Throw:
	g->m->throwing = 1;
	g->m->caughtsig = gp;
	runtime·startpanic();

	runtime·printf("%s\n", note);
	runtime·printf("PC=%X\n", ureg->ip);
	runtime·printf("\n");

	if(runtime·gotraceback(&crash)) {
		runtime·goroutineheader(gp);
		runtime·tracebacktrap(ureg->ip, ureg->sp, 0, gp);
		runtime·tracebackothers(gp);
		runtime·printf("\n");
		runtime·dumpregs(ureg);
	}
	
	if(crash)
		runtime·crash();

Exit:
	runtime·goexitsall(note);
	runtime·exits(note);
	return NDFLT; // not reached
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
	
	g->m->profilehz = hz;
}

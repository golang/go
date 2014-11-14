// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux
// +build power64 power64le

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"
#include "signal_GOOS_GOARCH.h"
#include "signals_GOOS.h"

void
runtime·dumpregs(Siginfo *info, void *ctxt)
{
	USED(info); USED(ctxt);
	runtime·printf("r0  %X\t", SIG_R0(info, ctxt));
	runtime·printf("r1  %X\n", SIG_R1(info, ctxt));
	runtime·printf("r2  %X\t", SIG_R2(info, ctxt));
	runtime·printf("r3  %X\n", SIG_R3(info, ctxt));
	runtime·printf("r4  %X\t", SIG_R4(info, ctxt));
	runtime·printf("r5  %X\n", SIG_R5(info, ctxt));
	runtime·printf("r6  %X\t", SIG_R6(info, ctxt));
	runtime·printf("r7  %X\n", SIG_R7(info, ctxt));
	runtime·printf("r8  %X\t", SIG_R8(info, ctxt));
	runtime·printf("r9  %X\n", SIG_R9(info, ctxt));
	runtime·printf("r10  %X\t", SIG_R10(info, ctxt));
	runtime·printf("r11  %X\n", SIG_R11(info, ctxt));
	runtime·printf("r12  %X\t", SIG_R12(info, ctxt));
	runtime·printf("r13  %X\n", SIG_R13(info, ctxt));
	runtime·printf("r14  %X\t", SIG_R14(info, ctxt));
	runtime·printf("r15  %X\n", SIG_R15(info, ctxt));
	runtime·printf("r16  %X\t", SIG_R16(info, ctxt));
	runtime·printf("r17  %X\n", SIG_R17(info, ctxt));
	runtime·printf("r18  %X\t", SIG_R18(info, ctxt));
	runtime·printf("r19  %X\n", SIG_R19(info, ctxt));
	runtime·printf("r20  %X\t", SIG_R20(info, ctxt));
	runtime·printf("r21  %X\n", SIG_R21(info, ctxt));
	runtime·printf("r22  %X\t", SIG_R22(info, ctxt));
	runtime·printf("r23  %X\n", SIG_R23(info, ctxt));
	runtime·printf("r24  %X\t", SIG_R24(info, ctxt));
	runtime·printf("r25  %X\n", SIG_R25(info, ctxt));
	runtime·printf("r26  %X\t", SIG_R26(info, ctxt));
	runtime·printf("r27  %X\n", SIG_R27(info, ctxt));
	runtime·printf("r28  %X\t", SIG_R28(info, ctxt));
	runtime·printf("r29  %X\n", SIG_R29(info, ctxt));
	runtime·printf("r30  %X\t", SIG_R30(info, ctxt));
	runtime·printf("r31  %X\n", SIG_R31(info, ctxt));
	runtime·printf("pc   %X\t", SIG_PC(info, ctxt));
	runtime·printf("ctr  %X\n", SIG_CTR(info, ctxt));
	runtime·printf("link %X\t", SIG_LINK(info, ctxt));
	runtime·printf("xer  %X\n", SIG_XER(info, ctxt));
	runtime·printf("ccr  %X\t", SIG_CCR(info, ctxt));
	runtime·printf("trap %X\n", SIG_TRAP(info, ctxt));
}

void
runtime·sighandler(int32 sig, Siginfo *info, void *ctxt, G *gp)
{
	SigTab *t;
	bool crash;

	if(sig == SIGPROF) {
		runtime·sigprof((uint8*)SIG_PC(info, ctxt), (uint8*)SIG_SP(info, ctxt), (uint8*)SIG_LINK(info, ctxt), gp, g->m);
		return;
	}
	t = &runtime·sigtab[sig];
	if(SIG_CODE0(info, ctxt) != SI_USER && (t->flags & SigPanic)) {
		// Make it look like a call to the signal func.
		// Have to pass arguments out of band since
		// augmenting the stack frame would break
		// the unwinding code.
		gp->sig = sig;
		gp->sigcode0 = SIG_CODE0(info, ctxt);
		gp->sigcode1 = SIG_FAULT(info, ctxt);
		gp->sigpc = SIG_PC(info, ctxt);

		// We arrange link, and pc to pretend the panicking
		// function calls sigpanic directly.
		// Always save LINK to stack so that panics in leaf
		// functions are correctly handled. This smashes
		// the stack frame but we're not going back there
		// anyway.
		SIG_SP(info, ctxt) -= sizeof(uintptr);
		*(uintptr*)SIG_SP(info, ctxt) = SIG_LINK(info, ctxt);
		// Don't bother saving PC if it's zero, which is
		// probably a call to a nil func: the old link register
		// is more useful in the stack trace.
		if(gp->sigpc != 0)
			SIG_LINK(info, ctxt) = gp->sigpc;
		// In case we are panicking from external C code
		SIG_R0(info, ctxt) = 0;
		SIG_R30(info, ctxt) = (uintptr)gp;
		SIG_PC(info, ctxt) = (uintptr)runtime·sigpanic;
		return;
	}

	if(SIG_CODE0(info, ctxt) == SI_USER || (t->flags & SigNotify))
		if(runtime·sigsend(sig))
			return;
	if(t->flags & SigKill)
		runtime·exit(2);
	if(!(t->flags & SigThrow))
		return;

	g->m->throwing = 1;
	g->m->caughtsig = gp;
	if(runtime·panicking)	// traceback already printed
		runtime·exit(2);
	runtime·panicking = 1;

	if(sig < 0 || sig >= NSIG)
		runtime·printf("Signal %d\n", sig);
	else
		runtime·printf("%s\n", runtime·sigtab[sig].name);

	runtime·printf("PC=%x\n", SIG_PC(info, ctxt));
	if(g->m->lockedg != nil && g->m->ncgo > 0 && gp == g->m->g0) {
		runtime·printf("signal arrived during cgo execution\n");
		gp = g->m->lockedg;
	}
	runtime·printf("\n");

	if(runtime·gotraceback(&crash)){
		runtime·goroutineheader(gp);
		runtime·tracebacktrap(SIG_PC(info, ctxt), SIG_SP(info, ctxt), SIG_LINK(info, ctxt), gp);
		runtime·tracebackothers(gp);
		runtime·printf("\n");
		runtime·dumpregs(info, ctxt);
	}
	
	if(crash)
		runtime·crash();

	runtime·exit(2);
}

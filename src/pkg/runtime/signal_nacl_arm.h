// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define SIG_REGS(ctxt) (((ExcContext*)(ctxt))->regs)

#define SIG_R0(info, ctxt) (SIG_REGS(ctxt).r0)
#define SIG_R1(info, ctxt) (SIG_REGS(ctxt).r1)
#define SIG_R2(info, ctxt) (SIG_REGS(ctxt).r2)
#define SIG_R3(info, ctxt) (SIG_REGS(ctxt).r3)
#define SIG_R4(info, ctxt) (SIG_REGS(ctxt).r4)
#define SIG_R5(info, ctxt) (SIG_REGS(ctxt).r5)
#define SIG_R6(info, ctxt) (SIG_REGS(ctxt).r6)
#define SIG_R7(info, ctxt) (SIG_REGS(ctxt).r7)
#define SIG_R8(info, ctxt) (SIG_REGS(ctxt).r8)
#define SIG_R9(info, ctxt) (SIG_REGS(ctxt).r9)
#define SIG_R10(info, ctxt) (SIG_REGS(ctxt).r10)
#define SIG_FP(info, ctxt) (SIG_REGS(ctxt).r11)
#define SIG_IP(info, ctxt) (SIG_REGS(ctxt).r12)
#define SIG_SP(info, ctxt) (SIG_REGS(ctxt).sp)
#define SIG_LR(info, ctxt) (SIG_REGS(ctxt).lr)
#define SIG_PC(info, ctxt) (SIG_REGS(ctxt).pc)
#define SIG_CPSR(info, ctxt) (SIG_REGS(ctxt).cpsr)
#define SIG_FAULT(info, ctxt) (~0)
#define SIG_TRAP(info, ctxt) (~0)
#define SIG_ERROR(info, ctxt) (~0)
#define SIG_OLDMASK(info, ctxt) (~0)
#define SIG_CODE0(info, ctxt) (~0)

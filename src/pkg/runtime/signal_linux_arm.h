// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define SIG_REGS(ctxt) (*((Sigcontext*)&((Ucontext*)(ctxt))->uc_mcontext))

#define SIG_R0(info, ctxt) (SIG_REGS(ctxt).arm_r0)
#define SIG_R1(info, ctxt) (SIG_REGS(ctxt).arm_r1)
#define SIG_R2(info, ctxt) (SIG_REGS(ctxt).arm_r2)
#define SIG_R3(info, ctxt) (SIG_REGS(ctxt).arm_r3)
#define SIG_R4(info, ctxt) (SIG_REGS(ctxt).arm_r4)
#define SIG_R5(info, ctxt) (SIG_REGS(ctxt).arm_r5)
#define SIG_R6(info, ctxt) (SIG_REGS(ctxt).arm_r6)
#define SIG_R7(info, ctxt) (SIG_REGS(ctxt).arm_r7)
#define SIG_R8(info, ctxt) (SIG_REGS(ctxt).arm_r8)
#define SIG_R9(info, ctxt) (SIG_REGS(ctxt).arm_r9)
#define SIG_R10(info, ctxt) (SIG_REGS(ctxt).arm_r10)
#define SIG_FP(info, ctxt) (SIG_REGS(ctxt).arm_fp)
#define SIG_IP(info, ctxt) (SIG_REGS(ctxt).arm_ip)
#define SIG_SP(info, ctxt) (SIG_REGS(ctxt).arm_sp)
#define SIG_LR(info, ctxt) (SIG_REGS(ctxt).arm_lr)
#define SIG_PC(info, ctxt) (SIG_REGS(ctxt).arm_pc)
#define SIG_CPSR(info, ctxt) (SIG_REGS(ctxt).arm_cpsr)
#define SIG_FAULT(info, ctxt) (SIG_REGS(ctxt).fault_address)
#define SIG_TRAP(info, ctxt) (SIG_REGS(ctxt).trap_no)
#define SIG_ERROR(info, ctxt) (SIG_REGS(ctxt).error_code)
#define SIG_OLDMASK(info, ctxt) (SIG_REGS(ctxt).oldmask)
#define SIG_CODE0(info, ctxt) ((uintptr)(info)->si_code)

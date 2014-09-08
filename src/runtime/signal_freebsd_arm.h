// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define SIG_REGS(ctxt) (((Ucontext*)(ctxt))->uc_mcontext)

#define SIG_R0(info, ctxt) (SIG_REGS(ctxt).__gregs[0])
#define SIG_R1(info, ctxt) (SIG_REGS(ctxt).__gregs[1])
#define SIG_R2(info, ctxt) (SIG_REGS(ctxt).__gregs[2])
#define SIG_R3(info, ctxt) (SIG_REGS(ctxt).__gregs[3])
#define SIG_R4(info, ctxt) (SIG_REGS(ctxt).__gregs[4])
#define SIG_R5(info, ctxt) (SIG_REGS(ctxt).__gregs[5])
#define SIG_R6(info, ctxt) (SIG_REGS(ctxt).__gregs[6])
#define SIG_R7(info, ctxt) (SIG_REGS(ctxt).__gregs[7])
#define SIG_R8(info, ctxt) (SIG_REGS(ctxt).__gregs[8])
#define SIG_R9(info, ctxt) (SIG_REGS(ctxt).__gregs[9])
#define SIG_R10(info, ctxt) (SIG_REGS(ctxt).__gregs[10])
#define SIG_FP(info, ctxt) (SIG_REGS(ctxt).__gregs[11])
#define SIG_IP(info, ctxt) (SIG_REGS(ctxt).__gregs[12])
#define SIG_SP(info, ctxt) (SIG_REGS(ctxt).__gregs[13])
#define SIG_LR(info, ctxt) (SIG_REGS(ctxt).__gregs[14])
#define SIG_PC(info, ctxt) (SIG_REGS(ctxt).__gregs[15])
#define SIG_CPSR(info, ctxt) (SIG_REGS(ctxt).__gregs[16])
#define SIG_FAULT(info, ctxt) ((uintptr)(info)->si_addr)
#define SIG_TRAP(info, ctxt) (0)
#define SIG_ERROR(info, ctxt) (0)
#define SIG_OLDMASK(info, ctxt) (0)
#define SIG_CODE0(info, ctxt) ((uintptr)(info)->si_code)

// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define SIG_REGS(ctxt) (((UcontextT*)(ctxt))->uc_mcontext)

#define SIG_R0(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_R0])
#define SIG_R1(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_R1])
#define SIG_R2(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_R2])
#define SIG_R3(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_R3])
#define SIG_R4(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_R4])
#define SIG_R5(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_R5])
#define SIG_R6(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_R6])
#define SIG_R7(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_R7])
#define SIG_R8(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_R8])
#define SIG_R9(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_R9])
#define SIG_R10(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_R10])
#define SIG_FP(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_R11])
#define SIG_IP(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_R12])
#define SIG_SP(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_R13])
#define SIG_LR(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_R14])
#define SIG_PC(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_R15])
#define SIG_CPSR(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_CPSR])
#define SIG_FAULT(info, ctxt) (*(uintptr*)&(info)->_reason[0])
#define SIG_TRAP(info, ctxt) (0)
#define SIG_ERROR(info, ctxt) (0)
#define SIG_OLDMASK(info, ctxt) (0)

#define SIG_CODE0(info, ctxt) ((info)->_code)
#define SIG_CODE1(info, ctxt) (*(uintptr*)&(info)->_reason[0])

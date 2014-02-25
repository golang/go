// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define SIG_REGS(ctxt) (((Ucontext*)(ctxt))->uc_mcontext)

#define SIG_RAX(info, ctxt) (SIG_REGS(ctxt).gregs[REG_RAX])
#define SIG_RBX(info, ctxt) (SIG_REGS(ctxt).gregs[REG_RBX])
#define SIG_RCX(info, ctxt) (SIG_REGS(ctxt).gregs[REG_RCX])
#define SIG_RDX(info, ctxt) (SIG_REGS(ctxt).gregs[REG_RDX])
#define SIG_RDI(info, ctxt) (SIG_REGS(ctxt).gregs[REG_RDI])
#define SIG_RSI(info, ctxt) (SIG_REGS(ctxt).gregs[REG_RSI])
#define SIG_RBP(info, ctxt) (SIG_REGS(ctxt).gregs[REG_RBP])
#define SIG_RSP(info, ctxt) (SIG_REGS(ctxt).gregs[REG_RSP])
#define SIG_R8(info, ctxt) (SIG_REGS(ctxt).gregs[REG_R8])
#define SIG_R9(info, ctxt) (SIG_REGS(ctxt).gregs[REG_R9])
#define SIG_R10(info, ctxt) (SIG_REGS(ctxt).gregs[REG_R10])
#define SIG_R11(info, ctxt) (SIG_REGS(ctxt).gregs[REG_R11])
#define SIG_R12(info, ctxt) (SIG_REGS(ctxt).gregs[REG_R12])
#define SIG_R13(info, ctxt) (SIG_REGS(ctxt).gregs[REG_R13])
#define SIG_R14(info, ctxt) (SIG_REGS(ctxt).gregs[REG_R14])
#define SIG_R15(info, ctxt) (SIG_REGS(ctxt).gregs[REG_R15])
#define SIG_RIP(info, ctxt) (SIG_REGS(ctxt).gregs[REG_RIP])
#define SIG_RFLAGS(info, ctxt) (SIG_REGS(ctxt).gregs[REG_RFLAGS])

#define SIG_CS(info, ctxt) (SIG_REGS(ctxt).gregs[REG_CS])
#define SIG_FS(info, ctxt) (SIG_REGS(ctxt).gregs[REG_FS])
#define SIG_GS(info, ctxt) (SIG_REGS(ctxt).gregs[REG_GS])

#define SIG_CODE0(info, ctxt) ((info)->si_code)
#define SIG_CODE1(info, ctxt) (*(uintptr*)&(info)->__data[0])

// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define SIG_REGS(ctxt) (((ExcContext*)(ctxt))->regs.regs64)

#define SIG_RAX(info, ctxt) (SIG_REGS(ctxt).rax)
#define SIG_RBX(info, ctxt) (SIG_REGS(ctxt).rbx)
#define SIG_RCX(info, ctxt) (SIG_REGS(ctxt).rcx)
#define SIG_RDX(info, ctxt) (SIG_REGS(ctxt).rdx)
#define SIG_RDI(info, ctxt) (SIG_REGS(ctxt).rdi)
#define SIG_RSI(info, ctxt) (SIG_REGS(ctxt).rsi)
#define SIG_RBP(info, ctxt) (SIG_REGS(ctxt).rbp)
#define SIG_RSP(info, ctxt) (SIG_REGS(ctxt).rsp)
#define SIG_R8(info, ctxt) (SIG_REGS(ctxt).r8)
#define SIG_R9(info, ctxt) (SIG_REGS(ctxt).r9)
#define SIG_R10(info, ctxt) (SIG_REGS(ctxt).r10)
#define SIG_R11(info, ctxt) (SIG_REGS(ctxt).r11)
#define SIG_R12(info, ctxt) (SIG_REGS(ctxt).r12)
#define SIG_R13(info, ctxt) (SIG_REGS(ctxt).r13)
#define SIG_R14(info, ctxt) (SIG_REGS(ctxt).r14)
#define SIG_R15(info, ctxt) (SIG_REGS(ctxt).r15)
#define SIG_RIP(info, ctxt) (SIG_REGS(ctxt).rip)
#define SIG_RFLAGS(info, ctxt) (SIG_REGS(ctxt).rflags)

#define SIG_CS(info, ctxt) (~0)
#define SIG_FS(info, ctxt) (~0)
#define SIG_GS(info, ctxt) (~0)

#define SIG_CODE0(info, ctxt) (~0)
#define SIG_CODE1(info, ctxt) (0)

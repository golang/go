// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define SIG_REGS(ctxt) (((Ucontext*)(ctxt))->uc_mcontext)

#define SIG_EAX(info, ctxt) (SIG_REGS(ctxt).mc_eax)
#define SIG_EBX(info, ctxt) (SIG_REGS(ctxt).mc_ebx)
#define SIG_ECX(info, ctxt) (SIG_REGS(ctxt).mc_ecx)
#define SIG_EDX(info, ctxt) (SIG_REGS(ctxt).mc_edx)
#define SIG_EDI(info, ctxt) (SIG_REGS(ctxt).mc_edi)
#define SIG_ESI(info, ctxt) (SIG_REGS(ctxt).mc_esi)
#define SIG_EBP(info, ctxt) (SIG_REGS(ctxt).mc_ebp)
#define SIG_ESP(info, ctxt) (SIG_REGS(ctxt).mc_esp)
#define SIG_EIP(info, ctxt) (SIG_REGS(ctxt).mc_eip)
#define SIG_EFLAGS(info, ctxt) (SIG_REGS(ctxt).mc_eflags)

#define SIG_CS(info, ctxt) (SIG_REGS(ctxt).mc_cs)
#define SIG_FS(info, ctxt) (SIG_REGS(ctxt).mc_fs)
#define SIG_GS(info, ctxt) (SIG_REGS(ctxt).mc_gs)

#define SIG_CODE0(info, ctxt) ((info)->si_code)
#define SIG_CODE1(info, ctxt) ((uintptr)(info)->si_addr)

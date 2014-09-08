// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define SIG_REGS(ctxt) (((Ucontext*)(ctxt))->uc_mcontext->ss)

#define SIG_EAX(info, ctxt) (SIG_REGS(ctxt).eax)
#define SIG_EBX(info, ctxt) (SIG_REGS(ctxt).ebx)
#define SIG_ECX(info, ctxt) (SIG_REGS(ctxt).ecx)
#define SIG_EDX(info, ctxt) (SIG_REGS(ctxt).edx)
#define SIG_EDI(info, ctxt) (SIG_REGS(ctxt).edi)
#define SIG_ESI(info, ctxt) (SIG_REGS(ctxt).esi)
#define SIG_EBP(info, ctxt) (SIG_REGS(ctxt).ebp)
#define SIG_ESP(info, ctxt) (SIG_REGS(ctxt).esp)
#define SIG_EIP(info, ctxt) (SIG_REGS(ctxt).eip)
#define SIG_EFLAGS(info, ctxt) (SIG_REGS(ctxt).eflags)

#define SIG_CS(info, ctxt) (SIG_REGS(ctxt).cs)
#define SIG_FS(info, ctxt) (SIG_REGS(ctxt).fs)
#define SIG_GS(info, ctxt) (SIG_REGS(ctxt).gs)

#define SIG_CODE0(info, ctxt) ((info)->si_code)
#define SIG_CODE1(info, ctxt) ((uintptr)(info)->si_addr)

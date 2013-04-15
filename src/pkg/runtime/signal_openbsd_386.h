// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define SIG_REGS(ctxt) (*(Sigcontext*)(ctxt))

#define SIG_EAX(info, ctxt) (SIG_REGS(ctxt).sc_eax)
#define SIG_EBX(info, ctxt) (SIG_REGS(ctxt).sc_ebx)
#define SIG_ECX(info, ctxt) (SIG_REGS(ctxt).sc_ecx)
#define SIG_EDX(info, ctxt) (SIG_REGS(ctxt).sc_edx)
#define SIG_EDI(info, ctxt) (SIG_REGS(ctxt).sc_edi)
#define SIG_ESI(info, ctxt) (SIG_REGS(ctxt).sc_esi)
#define SIG_EBP(info, ctxt) (SIG_REGS(ctxt).sc_ebp)
#define SIG_ESP(info, ctxt) (SIG_REGS(ctxt).sc_esp)
#define SIG_EIP(info, ctxt) (SIG_REGS(ctxt).sc_eip)
#define SIG_EFLAGS(info, ctxt) (SIG_REGS(ctxt).sc_eflags)

#define SIG_CS(info, ctxt) (SIG_REGS(ctxt).sc_cs)
#define SIG_FS(info, ctxt) (SIG_REGS(ctxt).sc_fs)
#define SIG_GS(info, ctxt) (SIG_REGS(ctxt).sc_gs)

#define SIG_CODE0(info, ctxt) ((info)->si_code)
#define SIG_CODE1(info, ctxt) (*(uintptr*)((byte*)info + 12))

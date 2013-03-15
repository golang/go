// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define SIG_REGS(ctxt) (((UcontextT*)(ctxt))->uc_mcontext)

#define SIG_EAX(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_EAX])
#define SIG_EBX(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_EBX])
#define SIG_ECX(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_ECX])
#define SIG_EDX(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_EDX])
#define SIG_EDI(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_EDI])
#define SIG_ESI(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_ESI])
#define SIG_EBP(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_EBP])
#define SIG_ESP(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_UESP])
#define SIG_EIP(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_EIP])
#define SIG_EFLAGS(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_EFL])

#define SIG_CS(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_CS])
#define SIG_FS(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_FS])
#define SIG_GS(info, ctxt) (SIG_REGS(ctxt).__gregs[REG_GS])

#define SIG_CODE0(info, ctxt) ((info)->_code)
#define SIG_CODE1(info, ctxt) (*(uintptr*)&(info)->_reason[0])

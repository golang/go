// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

type sigctxt struct {
	info *siginfo
	ctxt unsafe.Pointer
}

func (c *sigctxt) regs() *mcontext {
	return (*mcontext)(unsafe.Pointer(&(*ucontext)(c.ctxt).uc_mcontext))
}
func (c *sigctxt) rax() uint64     { return c.regs().mc_rax }
func (c *sigctxt) rbx() uint64     { return c.regs().mc_rbx }
func (c *sigctxt) rcx() uint64     { return c.regs().mc_rcx }
func (c *sigctxt) rdx() uint64     { return c.regs().mc_rdx }
func (c *sigctxt) rdi() uint64     { return c.regs().mc_rdi }
func (c *sigctxt) rsi() uint64     { return c.regs().mc_rsi }
func (c *sigctxt) rbp() uint64     { return c.regs().mc_rbp }
func (c *sigctxt) rsp() uint64     { return c.regs().mc_rsp }
func (c *sigctxt) r8() uint64      { return c.regs().mc_r8 }
func (c *sigctxt) r9() uint64      { return c.regs().mc_r9 }
func (c *sigctxt) r10() uint64     { return c.regs().mc_r10 }
func (c *sigctxt) r11() uint64     { return c.regs().mc_r11 }
func (c *sigctxt) r12() uint64     { return c.regs().mc_r12 }
func (c *sigctxt) r13() uint64     { return c.regs().mc_r13 }
func (c *sigctxt) r14() uint64     { return c.regs().mc_r14 }
func (c *sigctxt) r15() uint64     { return c.regs().mc_r15 }
func (c *sigctxt) rip() uint64     { return c.regs().mc_rip }
func (c *sigctxt) rflags() uint64  { return c.regs().mc_rflags }
func (c *sigctxt) cs() uint64      { return c.regs().mc_cs }
func (c *sigctxt) fs() uint64      { return c.regs().mc_ss }
func (c *sigctxt) gs() uint64      { return c.regs().mc_ss }
func (c *sigctxt) sigcode() uint64 { return uint64(c.info.si_code) }
func (c *sigctxt) sigaddr() uint64 { return c.info.si_addr }

func (c *sigctxt) set_rip(x uint64)     { c.regs().mc_rip = x }
func (c *sigctxt) set_rsp(x uint64)     { c.regs().mc_rsp = x }
func (c *sigctxt) set_sigcode(x uint64) { c.info.si_code = int32(x) }
func (c *sigctxt) set_sigaddr(x uint64) { c.info.si_addr = x }

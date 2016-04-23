// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

type sigctxt struct {
	info *siginfo
	ctxt unsafe.Pointer
}

func (c *sigctxt) regs() *mcontext { return &(*ucontext)(c.ctxt).uc_mcontext }
func (c *sigctxt) eax() uint32     { return c.regs().mc_eax }
func (c *sigctxt) ebx() uint32     { return c.regs().mc_ebx }
func (c *sigctxt) ecx() uint32     { return c.regs().mc_ecx }
func (c *sigctxt) edx() uint32     { return c.regs().mc_edx }
func (c *sigctxt) edi() uint32     { return c.regs().mc_edi }
func (c *sigctxt) esi() uint32     { return c.regs().mc_esi }
func (c *sigctxt) ebp() uint32     { return c.regs().mc_ebp }
func (c *sigctxt) esp() uint32     { return c.regs().mc_esp }
func (c *sigctxt) eip() uint32     { return c.regs().mc_eip }
func (c *sigctxt) eflags() uint32  { return c.regs().mc_eflags }
func (c *sigctxt) cs() uint32      { return c.regs().mc_cs }
func (c *sigctxt) fs() uint32      { return c.regs().mc_fs }
func (c *sigctxt) gs() uint32      { return c.regs().mc_gs }
func (c *sigctxt) sigcode() uint32 { return uint32(c.info.si_code) }
func (c *sigctxt) sigaddr() uint32 { return uint32(c.info.si_addr) }

func (c *sigctxt) set_eip(x uint32)     { c.regs().mc_eip = x }
func (c *sigctxt) set_esp(x uint32)     { c.regs().mc_esp = x }
func (c *sigctxt) set_sigcode(x uint32) { c.info.si_code = int32(x) }
func (c *sigctxt) set_sigaddr(x uint32) { c.info.si_addr = uintptr(x) }

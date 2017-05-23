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
func (c *sigctxt) r0() uint32      { return c.regs().__gregs[0] }
func (c *sigctxt) r1() uint32      { return c.regs().__gregs[1] }
func (c *sigctxt) r2() uint32      { return c.regs().__gregs[2] }
func (c *sigctxt) r3() uint32      { return c.regs().__gregs[3] }
func (c *sigctxt) r4() uint32      { return c.regs().__gregs[4] }
func (c *sigctxt) r5() uint32      { return c.regs().__gregs[5] }
func (c *sigctxt) r6() uint32      { return c.regs().__gregs[6] }
func (c *sigctxt) r7() uint32      { return c.regs().__gregs[7] }
func (c *sigctxt) r8() uint32      { return c.regs().__gregs[8] }
func (c *sigctxt) r9() uint32      { return c.regs().__gregs[9] }
func (c *sigctxt) r10() uint32     { return c.regs().__gregs[10] }
func (c *sigctxt) fp() uint32      { return c.regs().__gregs[11] }
func (c *sigctxt) ip() uint32      { return c.regs().__gregs[12] }
func (c *sigctxt) sp() uint32      { return c.regs().__gregs[13] }
func (c *sigctxt) lr() uint32      { return c.regs().__gregs[14] }
func (c *sigctxt) pc() uint32      { return c.regs().__gregs[15] }
func (c *sigctxt) cpsr() uint32    { return c.regs().__gregs[16] }
func (c *sigctxt) fault() uint32   { return uint32(c.info.si_addr) }
func (c *sigctxt) trap() uint32    { return 0 }
func (c *sigctxt) error() uint32   { return 0 }
func (c *sigctxt) oldmask() uint32 { return 0 }

func (c *sigctxt) sigcode() uint32 { return uint32(c.info.si_code) }
func (c *sigctxt) sigaddr() uint32 { return uint32(c.info.si_addr) }

func (c *sigctxt) set_pc(x uint32)  { c.regs().__gregs[15] = x }
func (c *sigctxt) set_sp(x uint32)  { c.regs().__gregs[13] = x }
func (c *sigctxt) set_lr(x uint32)  { c.regs().__gregs[14] = x }
func (c *sigctxt) set_r10(x uint32) { c.regs().__gregs[10] = x }

func (c *sigctxt) set_sigcode(x uint32) { c.info.si_code = int32(x) }
func (c *sigctxt) set_sigaddr(x uint32) {
	c.info.si_addr = uintptr(x)
}

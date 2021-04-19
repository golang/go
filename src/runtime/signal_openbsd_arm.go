// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

type sigctxt struct {
	info *siginfo
	ctxt unsafe.Pointer
}

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) regs() *sigcontext {
	return (*sigcontext)(c.ctxt)
}

func (c *sigctxt) r0() uint32  { return c.regs().sc_r0 }
func (c *sigctxt) r1() uint32  { return c.regs().sc_r1 }
func (c *sigctxt) r2() uint32  { return c.regs().sc_r2 }
func (c *sigctxt) r3() uint32  { return c.regs().sc_r3 }
func (c *sigctxt) r4() uint32  { return c.regs().sc_r4 }
func (c *sigctxt) r5() uint32  { return c.regs().sc_r5 }
func (c *sigctxt) r6() uint32  { return c.regs().sc_r6 }
func (c *sigctxt) r7() uint32  { return c.regs().sc_r7 }
func (c *sigctxt) r8() uint32  { return c.regs().sc_r8 }
func (c *sigctxt) r9() uint32  { return c.regs().sc_r9 }
func (c *sigctxt) r10() uint32 { return c.regs().sc_r10 }
func (c *sigctxt) fp() uint32  { return c.regs().sc_r11 }
func (c *sigctxt) ip() uint32  { return c.regs().sc_r12 }
func (c *sigctxt) sp() uint32  { return c.regs().sc_usr_sp }
func (c *sigctxt) lr() uint32  { return c.regs().sc_usr_lr }

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) pc() uint32 { return c.regs().sc_pc }

func (c *sigctxt) cpsr() uint32    { return c.regs().sc_spsr }
func (c *sigctxt) fault() uint32   { return c.sigaddr() }
func (c *sigctxt) trap() uint32    { return 0 }
func (c *sigctxt) error() uint32   { return 0 }
func (c *sigctxt) oldmask() uint32 { return 0 }

func (c *sigctxt) sigcode() uint32 { return uint32(c.info.si_code) }
func (c *sigctxt) sigaddr() uint32 {
	return *(*uint32)(add(unsafe.Pointer(c.info), 12))
}

func (c *sigctxt) set_pc(x uint32)  { c.regs().sc_pc = x }
func (c *sigctxt) set_sp(x uint32)  { c.regs().sc_usr_sp = x }
func (c *sigctxt) set_lr(x uint32)  { c.regs().sc_usr_lr = x }
func (c *sigctxt) set_r10(x uint32) { c.regs().sc_r10 = x }

func (c *sigctxt) set_sigcode(x uint32) { c.info.si_code = int32(x) }
func (c *sigctxt) set_sigaddr(x uint32) {
	*(*uint32)(add(unsafe.Pointer(c.info), 12)) = x
}

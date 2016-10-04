// Copyright 2014 The Go Authors. All rights reserved.
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
func (c *sigctxt) regs() *regs32 { return &(*ucontext)(c.ctxt).uc_mcontext.ss }

func (c *sigctxt) r0() uint32  { return c.regs().r[0] }
func (c *sigctxt) r1() uint32  { return c.regs().r[1] }
func (c *sigctxt) r2() uint32  { return c.regs().r[2] }
func (c *sigctxt) r3() uint32  { return c.regs().r[3] }
func (c *sigctxt) r4() uint32  { return c.regs().r[4] }
func (c *sigctxt) r5() uint32  { return c.regs().r[5] }
func (c *sigctxt) r6() uint32  { return c.regs().r[6] }
func (c *sigctxt) r7() uint32  { return c.regs().r[7] }
func (c *sigctxt) r8() uint32  { return c.regs().r[8] }
func (c *sigctxt) r9() uint32  { return c.regs().r[9] }
func (c *sigctxt) r10() uint32 { return c.regs().r[10] }
func (c *sigctxt) fp() uint32  { return c.regs().r[11] }
func (c *sigctxt) ip() uint32  { return c.regs().r[12] }
func (c *sigctxt) sp() uint32  { return c.regs().sp }
func (c *sigctxt) lr() uint32  { return c.regs().lr }

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) pc() uint32 { return c.regs().pc }

func (c *sigctxt) cpsr() uint32    { return c.regs().cpsr }
func (c *sigctxt) fault() uint32   { return c.info.si_addr }
func (c *sigctxt) sigcode() uint32 { return uint32(c.info.si_code) }
func (c *sigctxt) trap() uint32    { return 0 }
func (c *sigctxt) error() uint32   { return 0 }
func (c *sigctxt) oldmask() uint32 { return 0 }

func (c *sigctxt) set_pc(x uint32)  { c.regs().pc = x }
func (c *sigctxt) set_sp(x uint32)  { c.regs().sp = x }
func (c *sigctxt) set_lr(x uint32)  { c.regs().lr = x }
func (c *sigctxt) set_r10(x uint32) { c.regs().r[10] = x }

func (c *sigctxt) set_sigcode(x uint32) { c.info.si_code = int32(x) }
func (c *sigctxt) set_sigaddr(x uint32) { c.info.si_addr = x }

func (c *sigctxt) fixsigcode(sig uint32) {
	switch sig {
	case _SIGTRAP:
		// OS X sets c.sigcode() == TRAP_BRKPT unconditionally for all SIGTRAPs,
		// leaving no way to distinguish a breakpoint-induced SIGTRAP
		// from an asynchronous signal SIGTRAP.
		// They all look breakpoint-induced by default.
		// Try looking at the code to see if it's a breakpoint.
		// The assumption is that we're very unlikely to get an
		// asynchronous SIGTRAP at just the moment that the
		// PC started to point at unmapped memory.
		pc := uintptr(c.pc())
		// OS X will leave the pc just after the instruction.
		code := (*uint32)(unsafe.Pointer(pc - 4))
		if *code != 0xe7f001f0 {
			// SIGTRAP on something other than breakpoint.
			c.set_sigcode(_SI_USER)
		}
	}
}

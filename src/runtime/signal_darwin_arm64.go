// Copyright 2015 The Go Authors. All rights reserved.
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
func (c *sigctxt) regs() *regs64 { return &(*ucontext)(c.ctxt).uc_mcontext.ss }

func (c *sigctxt) r0() uint64  { return c.regs().x[0] }
func (c *sigctxt) r1() uint64  { return c.regs().x[1] }
func (c *sigctxt) r2() uint64  { return c.regs().x[2] }
func (c *sigctxt) r3() uint64  { return c.regs().x[3] }
func (c *sigctxt) r4() uint64  { return c.regs().x[4] }
func (c *sigctxt) r5() uint64  { return c.regs().x[5] }
func (c *sigctxt) r6() uint64  { return c.regs().x[6] }
func (c *sigctxt) r7() uint64  { return c.regs().x[7] }
func (c *sigctxt) r8() uint64  { return c.regs().x[8] }
func (c *sigctxt) r9() uint64  { return c.regs().x[9] }
func (c *sigctxt) r10() uint64 { return c.regs().x[10] }
func (c *sigctxt) r11() uint64 { return c.regs().x[11] }
func (c *sigctxt) r12() uint64 { return c.regs().x[12] }
func (c *sigctxt) r13() uint64 { return c.regs().x[13] }
func (c *sigctxt) r14() uint64 { return c.regs().x[14] }
func (c *sigctxt) r15() uint64 { return c.regs().x[15] }
func (c *sigctxt) r16() uint64 { return c.regs().x[16] }
func (c *sigctxt) r17() uint64 { return c.regs().x[17] }
func (c *sigctxt) r18() uint64 { return c.regs().x[18] }
func (c *sigctxt) r19() uint64 { return c.regs().x[19] }
func (c *sigctxt) r20() uint64 { return c.regs().x[20] }
func (c *sigctxt) r21() uint64 { return c.regs().x[21] }
func (c *sigctxt) r22() uint64 { return c.regs().x[22] }
func (c *sigctxt) r23() uint64 { return c.regs().x[23] }
func (c *sigctxt) r24() uint64 { return c.regs().x[24] }
func (c *sigctxt) r25() uint64 { return c.regs().x[25] }
func (c *sigctxt) r26() uint64 { return c.regs().x[26] }
func (c *sigctxt) r27() uint64 { return c.regs().x[27] }
func (c *sigctxt) r28() uint64 { return c.regs().x[28] }
func (c *sigctxt) r29() uint64 { return c.regs().fp }
func (c *sigctxt) lr() uint64  { return c.regs().lr }
func (c *sigctxt) sp() uint64  { return c.regs().sp }

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) pc() uint64 { return c.regs().pc }

func (c *sigctxt) fault() uint64 { return uint64(uintptr(unsafe.Pointer(c.info.si_addr))) }

func (c *sigctxt) sigcode() uint64 { return uint64(c.info.si_code) }
func (c *sigctxt) sigaddr() uint64 { return uint64(uintptr(unsafe.Pointer(c.info.si_addr))) }

func (c *sigctxt) set_pc(x uint64)  { c.regs().pc = x }
func (c *sigctxt) set_sp(x uint64)  { c.regs().sp = x }
func (c *sigctxt) set_lr(x uint64)  { c.regs().lr = x }
func (c *sigctxt) set_r28(x uint64) { c.regs().x[28] = x }

func (c *sigctxt) set_sigcode(x uint64) { c.info.si_code = int32(x) }
func (c *sigctxt) set_sigaddr(x uint64) {
	c.info.si_addr = (*byte)(unsafe.Pointer(uintptr(x)))
}

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
		if *code != 0xd4200000 {
			// SIGTRAP on something other than breakpoint.
			c.set_sigcode(_SI_USER)
		}
	}
}

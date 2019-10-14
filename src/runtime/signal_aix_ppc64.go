/// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix

package runtime

import (
	"runtime/internal/sys"
	"unsafe"
)

type sigctxt struct {
	info *siginfo
	ctxt unsafe.Pointer
}

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) regs() *context64 { return &(*ucontext)(c.ctxt).uc_mcontext }

func (c *sigctxt) r0() uint64  { return c.regs().gpr[0] }
func (c *sigctxt) r1() uint64  { return c.regs().gpr[1] }
func (c *sigctxt) r2() uint64  { return c.regs().gpr[2] }
func (c *sigctxt) r3() uint64  { return c.regs().gpr[3] }
func (c *sigctxt) r4() uint64  { return c.regs().gpr[4] }
func (c *sigctxt) r5() uint64  { return c.regs().gpr[5] }
func (c *sigctxt) r6() uint64  { return c.regs().gpr[6] }
func (c *sigctxt) r7() uint64  { return c.regs().gpr[7] }
func (c *sigctxt) r8() uint64  { return c.regs().gpr[8] }
func (c *sigctxt) r9() uint64  { return c.regs().gpr[9] }
func (c *sigctxt) r10() uint64 { return c.regs().gpr[10] }
func (c *sigctxt) r11() uint64 { return c.regs().gpr[11] }
func (c *sigctxt) r12() uint64 { return c.regs().gpr[12] }
func (c *sigctxt) r13() uint64 { return c.regs().gpr[13] }
func (c *sigctxt) r14() uint64 { return c.regs().gpr[14] }
func (c *sigctxt) r15() uint64 { return c.regs().gpr[15] }
func (c *sigctxt) r16() uint64 { return c.regs().gpr[16] }
func (c *sigctxt) r17() uint64 { return c.regs().gpr[17] }
func (c *sigctxt) r18() uint64 { return c.regs().gpr[18] }
func (c *sigctxt) r19() uint64 { return c.regs().gpr[19] }
func (c *sigctxt) r20() uint64 { return c.regs().gpr[20] }
func (c *sigctxt) r21() uint64 { return c.regs().gpr[21] }
func (c *sigctxt) r22() uint64 { return c.regs().gpr[22] }
func (c *sigctxt) r23() uint64 { return c.regs().gpr[23] }
func (c *sigctxt) r24() uint64 { return c.regs().gpr[24] }
func (c *sigctxt) r25() uint64 { return c.regs().gpr[25] }
func (c *sigctxt) r26() uint64 { return c.regs().gpr[26] }
func (c *sigctxt) r27() uint64 { return c.regs().gpr[27] }
func (c *sigctxt) r28() uint64 { return c.regs().gpr[28] }
func (c *sigctxt) r29() uint64 { return c.regs().gpr[29] }
func (c *sigctxt) r30() uint64 { return c.regs().gpr[30] }
func (c *sigctxt) r31() uint64 { return c.regs().gpr[31] }
func (c *sigctxt) sp() uint64  { return c.regs().gpr[1] }

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) pc() uint64 { return c.regs().iar }

func (c *sigctxt) ctr() uint64    { return c.regs().ctr }
func (c *sigctxt) link() uint64   { return c.regs().lr }
func (c *sigctxt) xer() uint32    { return c.regs().xer }
func (c *sigctxt) ccr() uint32    { return c.regs().cr }
func (c *sigctxt) fpscr() uint32  { return c.regs().fpscr }
func (c *sigctxt) fpscrx() uint32 { return c.regs().fpscrx }

// TODO(aix): find trap equivalent
func (c *sigctxt) trap() uint32 { return 0x0 }

func (c *sigctxt) sigcode() uint32 { return uint32(c.info.si_code) }
func (c *sigctxt) sigaddr() uint64 { return uint64(c.info.si_addr) }
func (c *sigctxt) fault() uintptr  { return uintptr(c.sigaddr()) }

func (c *sigctxt) set_r0(x uint64)   { c.regs().gpr[0] = x }
func (c *sigctxt) set_r12(x uint64)  { c.regs().gpr[12] = x }
func (c *sigctxt) set_r30(x uint64)  { c.regs().gpr[30] = x }
func (c *sigctxt) set_pc(x uint64)   { c.regs().iar = x }
func (c *sigctxt) set_sp(x uint64)   { c.regs().gpr[1] = x }
func (c *sigctxt) set_link(x uint64) { c.regs().lr = x }

func (c *sigctxt) set_sigcode(x uint32) { c.info.si_code = int32(x) }
func (c *sigctxt) set_sigaddr(x uint64) {
	*(*uintptr)(add(unsafe.Pointer(c.info), 2*sys.PtrSize)) = uintptr(x)
}

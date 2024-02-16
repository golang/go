// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/goarch"
	"unsafe"
)

type sigctxt struct {
	info *siginfo
	ctxt unsafe.Pointer
}

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) regs() *sigcontext {
	return (*sigcontext)(c.ctxt)
}

func (c *sigctxt) r0() uint64  { return c.regs().sc_reg[0] }
func (c *sigctxt) r1() uint64  { return c.regs().sc_reg[1] }
func (c *sigctxt) r2() uint64  { return c.regs().sc_reg[2] }
func (c *sigctxt) r3() uint64  { return c.regs().sc_reg[3] }
func (c *sigctxt) r4() uint64  { return c.regs().sc_reg[4] }
func (c *sigctxt) r5() uint64  { return c.regs().sc_reg[5] }
func (c *sigctxt) r6() uint64  { return c.regs().sc_reg[6] }
func (c *sigctxt) r7() uint64  { return c.regs().sc_reg[7] }
func (c *sigctxt) r8() uint64  { return c.regs().sc_reg[8] }
func (c *sigctxt) r9() uint64  { return c.regs().sc_reg[9] }
func (c *sigctxt) r10() uint64 { return c.regs().sc_reg[10] }
func (c *sigctxt) r11() uint64 { return c.regs().sc_reg[11] }
func (c *sigctxt) r12() uint64 { return c.regs().sc_reg[12] }
func (c *sigctxt) r13() uint64 { return c.regs().sc_reg[13] }
func (c *sigctxt) r14() uint64 { return c.regs().sc_reg[14] }
func (c *sigctxt) r15() uint64 { return c.regs().sc_reg[15] }
func (c *sigctxt) r16() uint64 { return c.regs().sc_reg[16] }
func (c *sigctxt) r17() uint64 { return c.regs().sc_reg[17] }
func (c *sigctxt) r18() uint64 { return c.regs().sc_reg[18] }
func (c *sigctxt) r19() uint64 { return c.regs().sc_reg[19] }
func (c *sigctxt) r20() uint64 { return c.regs().sc_reg[20] }
func (c *sigctxt) r21() uint64 { return c.regs().sc_reg[21] }
func (c *sigctxt) r22() uint64 { return c.regs().sc_reg[22] }
func (c *sigctxt) r23() uint64 { return c.regs().sc_reg[23] }
func (c *sigctxt) r24() uint64 { return c.regs().sc_reg[24] }
func (c *sigctxt) r25() uint64 { return c.regs().sc_reg[25] }
func (c *sigctxt) r26() uint64 { return c.regs().sc_reg[26] }
func (c *sigctxt) r27() uint64 { return c.regs().sc_reg[27] }
func (c *sigctxt) r28() uint64 { return c.regs().sc_reg[28] }
func (c *sigctxt) r29() uint64 { return c.regs().sc_reg[29] }
func (c *sigctxt) r30() uint64 { return c.regs().sc_reg[30] }
func (c *sigctxt) r31() uint64 { return c.regs().sc_reg[31] }
func (c *sigctxt) sp() uint64  { return c.regs().sc_reg[1] }

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) pc() uint64 { return c.regs().sc_pc }

func (c *sigctxt) trap() uint64 { return 0 /* XXX - c.regs().trap */ }
func (c *sigctxt) ctr() uint64  { return c.regs().sc_ctr }
func (c *sigctxt) link() uint64 { return c.regs().sc_lr }
func (c *sigctxt) xer() uint64  { return c.regs().sc_xer }
func (c *sigctxt) ccr() uint64  { return c.regs().sc_cr }

func (c *sigctxt) sigcode() uint32 { return uint32(c.info.si_code) }
func (c *sigctxt) sigaddr() uint64 {
	return *(*uint64)(add(unsafe.Pointer(c.info), 16))
}
func (c *sigctxt) fault() uintptr { return uintptr(c.sigaddr()) }

func (c *sigctxt) set_r0(x uint64)   { c.regs().sc_reg[0] = x }
func (c *sigctxt) set_r12(x uint64)  { c.regs().sc_reg[12] = x }
func (c *sigctxt) set_r30(x uint64)  { c.regs().sc_reg[30] = x }
func (c *sigctxt) set_pc(x uint64)   { c.regs().sc_pc = x }
func (c *sigctxt) set_sp(x uint64)   { c.regs().sc_reg[1] = x }
func (c *sigctxt) set_link(x uint64) { c.regs().sc_lr = x }

func (c *sigctxt) set_sigcode(x uint32) { c.info.si_code = int32(x) }
func (c *sigctxt) set_sigaddr(x uint64) {
	*(*uintptr)(add(unsafe.Pointer(c.info), 2*goarch.PtrSize)) = uintptr(x)
}

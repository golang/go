// Copyright 2019 The Go Authors. All rights reserved.
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
func (c *sigctxt) regs() *mcontextt {
	return (*mcontextt)(unsafe.Pointer(&(*ucontextt)(c.ctxt).uc_mcontext))
}

func (c *sigctxt) r0() uint64  { return c.regs().__gregs[_REG_X0] }
func (c *sigctxt) r1() uint64  { return c.regs().__gregs[_REG_X1] }
func (c *sigctxt) r2() uint64  { return c.regs().__gregs[_REG_X2] }
func (c *sigctxt) r3() uint64  { return c.regs().__gregs[_REG_X3] }
func (c *sigctxt) r4() uint64  { return c.regs().__gregs[_REG_X4] }
func (c *sigctxt) r5() uint64  { return c.regs().__gregs[_REG_X5] }
func (c *sigctxt) r6() uint64  { return c.regs().__gregs[_REG_X6] }
func (c *sigctxt) r7() uint64  { return c.regs().__gregs[_REG_X7] }
func (c *sigctxt) r8() uint64  { return c.regs().__gregs[_REG_X8] }
func (c *sigctxt) r9() uint64  { return c.regs().__gregs[_REG_X9] }
func (c *sigctxt) r10() uint64 { return c.regs().__gregs[_REG_X10] }
func (c *sigctxt) r11() uint64 { return c.regs().__gregs[_REG_X11] }
func (c *sigctxt) r12() uint64 { return c.regs().__gregs[_REG_X12] }
func (c *sigctxt) r13() uint64 { return c.regs().__gregs[_REG_X13] }
func (c *sigctxt) r14() uint64 { return c.regs().__gregs[_REG_X14] }
func (c *sigctxt) r15() uint64 { return c.regs().__gregs[_REG_X15] }
func (c *sigctxt) r16() uint64 { return c.regs().__gregs[_REG_X16] }
func (c *sigctxt) r17() uint64 { return c.regs().__gregs[_REG_X17] }
func (c *sigctxt) r18() uint64 { return c.regs().__gregs[_REG_X18] }
func (c *sigctxt) r19() uint64 { return c.regs().__gregs[_REG_X19] }
func (c *sigctxt) r20() uint64 { return c.regs().__gregs[_REG_X20] }
func (c *sigctxt) r21() uint64 { return c.regs().__gregs[_REG_X21] }
func (c *sigctxt) r22() uint64 { return c.regs().__gregs[_REG_X22] }
func (c *sigctxt) r23() uint64 { return c.regs().__gregs[_REG_X23] }
func (c *sigctxt) r24() uint64 { return c.regs().__gregs[_REG_X24] }
func (c *sigctxt) r25() uint64 { return c.regs().__gregs[_REG_X25] }
func (c *sigctxt) r26() uint64 { return c.regs().__gregs[_REG_X26] }
func (c *sigctxt) r27() uint64 { return c.regs().__gregs[_REG_X27] }
func (c *sigctxt) r28() uint64 { return c.regs().__gregs[_REG_X28] }
func (c *sigctxt) r29() uint64 { return c.regs().__gregs[_REG_X29] }
func (c *sigctxt) lr() uint64  { return c.regs().__gregs[_REG_X30] }
func (c *sigctxt) sp() uint64  { return c.regs().__gregs[_REG_X31] }

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) pc() uint64 { return c.regs().__gregs[_REG_ELR] }

func (c *sigctxt) fault() uintptr  { return uintptr(c.info._reason) }
func (c *sigctxt) trap() uint64    { return 0 }
func (c *sigctxt) error() uint64   { return 0 }
func (c *sigctxt) oldmask() uint64 { return 0 }

func (c *sigctxt) sigcode() uint64 { return uint64(c.info._code) }
func (c *sigctxt) sigaddr() uint64 { return uint64(c.info._reason) }

func (c *sigctxt) set_pc(x uint64)  { c.regs().__gregs[_REG_ELR] = x }
func (c *sigctxt) set_sp(x uint64)  { c.regs().__gregs[_REG_X31] = x }
func (c *sigctxt) set_lr(x uint64)  { c.regs().__gregs[_REG_X30] = x }
func (c *sigctxt) set_r28(x uint64) { c.regs().__gregs[_REG_X28] = x }

func (c *sigctxt) set_sigcode(x uint64) { c.info._code = int32(x) }
func (c *sigctxt) set_sigaddr(x uint64) {
	c.info._reason = uintptr(x)
}

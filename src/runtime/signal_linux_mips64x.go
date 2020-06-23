// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux
// +build mips64 mips64le

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
func (c *sigctxt) regs() *sigcontext { return &(*ucontext)(c.ctxt).uc_mcontext }

func (c *sigctxt) r0() uint64  { return c.regs().sc_regs[0] }
func (c *sigctxt) r1() uint64  { return c.regs().sc_regs[1] }
func (c *sigctxt) r2() uint64  { return c.regs().sc_regs[2] }
func (c *sigctxt) r3() uint64  { return c.regs().sc_regs[3] }
func (c *sigctxt) r4() uint64  { return c.regs().sc_regs[4] }
func (c *sigctxt) r5() uint64  { return c.regs().sc_regs[5] }
func (c *sigctxt) r6() uint64  { return c.regs().sc_regs[6] }
func (c *sigctxt) r7() uint64  { return c.regs().sc_regs[7] }
func (c *sigctxt) r8() uint64  { return c.regs().sc_regs[8] }
func (c *sigctxt) r9() uint64  { return c.regs().sc_regs[9] }
func (c *sigctxt) r10() uint64 { return c.regs().sc_regs[10] }
func (c *sigctxt) r11() uint64 { return c.regs().sc_regs[11] }
func (c *sigctxt) r12() uint64 { return c.regs().sc_regs[12] }
func (c *sigctxt) r13() uint64 { return c.regs().sc_regs[13] }
func (c *sigctxt) r14() uint64 { return c.regs().sc_regs[14] }
func (c *sigctxt) r15() uint64 { return c.regs().sc_regs[15] }
func (c *sigctxt) r16() uint64 { return c.regs().sc_regs[16] }
func (c *sigctxt) r17() uint64 { return c.regs().sc_regs[17] }
func (c *sigctxt) r18() uint64 { return c.regs().sc_regs[18] }
func (c *sigctxt) r19() uint64 { return c.regs().sc_regs[19] }
func (c *sigctxt) r20() uint64 { return c.regs().sc_regs[20] }
func (c *sigctxt) r21() uint64 { return c.regs().sc_regs[21] }
func (c *sigctxt) r22() uint64 { return c.regs().sc_regs[22] }
func (c *sigctxt) r23() uint64 { return c.regs().sc_regs[23] }
func (c *sigctxt) r24() uint64 { return c.regs().sc_regs[24] }
func (c *sigctxt) r25() uint64 { return c.regs().sc_regs[25] }
func (c *sigctxt) r26() uint64 { return c.regs().sc_regs[26] }
func (c *sigctxt) r27() uint64 { return c.regs().sc_regs[27] }
func (c *sigctxt) r28() uint64 { return c.regs().sc_regs[28] }
func (c *sigctxt) r29() uint64 { return c.regs().sc_regs[29] }
func (c *sigctxt) r30() uint64 { return c.regs().sc_regs[30] }
func (c *sigctxt) r31() uint64 { return c.regs().sc_regs[31] }
func (c *sigctxt) sp() uint64  { return c.regs().sc_regs[29] }

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) pc() uint64 { return c.regs().sc_pc }

func (c *sigctxt) link() uint64 { return c.regs().sc_regs[31] }
func (c *sigctxt) lo() uint64   { return c.regs().sc_mdlo }
func (c *sigctxt) hi() uint64   { return c.regs().sc_mdhi }

func (c *sigctxt) sigcode() uint32 { return uint32(c.info.si_code) }
func (c *sigctxt) sigaddr() uint64 { return c.info.si_addr }

func (c *sigctxt) set_r28(x uint64)  { c.regs().sc_regs[28] = x }
func (c *sigctxt) set_r30(x uint64)  { c.regs().sc_regs[30] = x }
func (c *sigctxt) set_pc(x uint64)   { c.regs().sc_pc = x }
func (c *sigctxt) set_sp(x uint64)   { c.regs().sc_regs[29] = x }
func (c *sigctxt) set_link(x uint64) { c.regs().sc_regs[31] = x }

func (c *sigctxt) set_sigcode(x uint32) { c.info.si_code = int32(x) }
func (c *sigctxt) set_sigaddr(x uint64) {
	*(*uintptr)(add(unsafe.Pointer(c.info), 2*sys.PtrSize)) = uintptr(x)
}

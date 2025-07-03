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
func (c *sigctxt) regs() *sigcontext {
	return (*sigcontext)(c.ctxt)
}

func (c *sigctxt) r0() uint64  { return (uint64)(c.regs().sc_x[0]) }
func (c *sigctxt) r1() uint64  { return (uint64)(c.regs().sc_x[1]) }
func (c *sigctxt) r2() uint64  { return (uint64)(c.regs().sc_x[2]) }
func (c *sigctxt) r3() uint64  { return (uint64)(c.regs().sc_x[3]) }
func (c *sigctxt) r4() uint64  { return (uint64)(c.regs().sc_x[4]) }
func (c *sigctxt) r5() uint64  { return (uint64)(c.regs().sc_x[5]) }
func (c *sigctxt) r6() uint64  { return (uint64)(c.regs().sc_x[6]) }
func (c *sigctxt) r7() uint64  { return (uint64)(c.regs().sc_x[7]) }
func (c *sigctxt) r8() uint64  { return (uint64)(c.regs().sc_x[8]) }
func (c *sigctxt) r9() uint64  { return (uint64)(c.regs().sc_x[9]) }
func (c *sigctxt) r10() uint64 { return (uint64)(c.regs().sc_x[10]) }
func (c *sigctxt) r11() uint64 { return (uint64)(c.regs().sc_x[11]) }
func (c *sigctxt) r12() uint64 { return (uint64)(c.regs().sc_x[12]) }
func (c *sigctxt) r13() uint64 { return (uint64)(c.regs().sc_x[13]) }
func (c *sigctxt) r14() uint64 { return (uint64)(c.regs().sc_x[14]) }
func (c *sigctxt) r15() uint64 { return (uint64)(c.regs().sc_x[15]) }
func (c *sigctxt) r16() uint64 { return (uint64)(c.regs().sc_x[16]) }
func (c *sigctxt) r17() uint64 { return (uint64)(c.regs().sc_x[17]) }
func (c *sigctxt) r18() uint64 { return (uint64)(c.regs().sc_x[18]) }
func (c *sigctxt) r19() uint64 { return (uint64)(c.regs().sc_x[19]) }
func (c *sigctxt) r20() uint64 { return (uint64)(c.regs().sc_x[20]) }
func (c *sigctxt) r21() uint64 { return (uint64)(c.regs().sc_x[21]) }
func (c *sigctxt) r22() uint64 { return (uint64)(c.regs().sc_x[22]) }
func (c *sigctxt) r23() uint64 { return (uint64)(c.regs().sc_x[23]) }
func (c *sigctxt) r24() uint64 { return (uint64)(c.regs().sc_x[24]) }
func (c *sigctxt) r25() uint64 { return (uint64)(c.regs().sc_x[25]) }
func (c *sigctxt) r26() uint64 { return (uint64)(c.regs().sc_x[26]) }
func (c *sigctxt) r27() uint64 { return (uint64)(c.regs().sc_x[27]) }
func (c *sigctxt) r28() uint64 { return (uint64)(c.regs().sc_x[28]) }
func (c *sigctxt) r29() uint64 { return (uint64)(c.regs().sc_x[29]) }
func (c *sigctxt) lr() uint64  { return (uint64)(c.regs().sc_lr) }
func (c *sigctxt) sp() uint64  { return (uint64)(c.regs().sc_sp) }

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) rip() uint64 { return (uint64)(c.regs().sc_lr) } /* XXX */

func (c *sigctxt) fault() uintptr  { return uintptr(c.sigaddr()) }
func (c *sigctxt) sigcode() uint64 { return uint64(c.info.si_code) }
func (c *sigctxt) sigaddr() uint64 {
	return *(*uint64)(add(unsafe.Pointer(c.info), 16))
}

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) pc() uint64 { return uint64(c.regs().sc_elr) }

func (c *sigctxt) set_pc(x uint64)  { c.regs().sc_elr = uintptr(x) }
func (c *sigctxt) set_sp(x uint64)  { c.regs().sc_sp = uintptr(x) }
func (c *sigctxt) set_lr(x uint64)  { c.regs().sc_lr = uintptr(x) }
func (c *sigctxt) set_r28(x uint64) { c.regs().sc_x[28] = uintptr(x) }

func (c *sigctxt) set_sigcode(x uint64) { c.info.si_code = int32(x) }
func (c *sigctxt) set_sigaddr(x uint64) {
	*(*uint64)(add(unsafe.Pointer(c.info), 16)) = x
}

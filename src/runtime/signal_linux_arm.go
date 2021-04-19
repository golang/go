// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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

func (c *sigctxt) r0() uint32  { return c.regs().r0 }
func (c *sigctxt) r1() uint32  { return c.regs().r1 }
func (c *sigctxt) r2() uint32  { return c.regs().r2 }
func (c *sigctxt) r3() uint32  { return c.regs().r3 }
func (c *sigctxt) r4() uint32  { return c.regs().r4 }
func (c *sigctxt) r5() uint32  { return c.regs().r5 }
func (c *sigctxt) r6() uint32  { return c.regs().r6 }
func (c *sigctxt) r7() uint32  { return c.regs().r7 }
func (c *sigctxt) r8() uint32  { return c.regs().r8 }
func (c *sigctxt) r9() uint32  { return c.regs().r9 }
func (c *sigctxt) r10() uint32 { return c.regs().r10 }
func (c *sigctxt) fp() uint32  { return c.regs().fp }
func (c *sigctxt) ip() uint32  { return c.regs().ip }
func (c *sigctxt) sp() uint32  { return c.regs().sp }
func (c *sigctxt) lr() uint32  { return c.regs().lr }

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) pc() uint32 { return c.regs().pc }

func (c *sigctxt) cpsr() uint32    { return c.regs().cpsr }
func (c *sigctxt) fault() uint32   { return c.regs().fault_address }
func (c *sigctxt) trap() uint32    { return c.regs().trap_no }
func (c *sigctxt) error() uint32   { return c.regs().error_code }
func (c *sigctxt) oldmask() uint32 { return c.regs().oldmask }

func (c *sigctxt) sigcode() uint32 { return uint32(c.info.si_code) }
func (c *sigctxt) sigaddr() uint32 { return c.info.si_addr }

func (c *sigctxt) set_pc(x uint32)  { c.regs().pc = x }
func (c *sigctxt) set_sp(x uint32)  { c.regs().sp = x }
func (c *sigctxt) set_lr(x uint32)  { c.regs().lr = x }
func (c *sigctxt) set_r10(x uint32) { c.regs().r10 = x }

func (c *sigctxt) set_sigcode(x uint32) { c.info.si_code = int32(x) }
func (c *sigctxt) set_sigaddr(x uint32) {
	*(*uintptr)(add(unsafe.Pointer(c.info), 2*sys.PtrSize)) = uintptr(x)
}

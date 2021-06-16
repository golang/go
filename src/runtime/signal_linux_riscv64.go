// Copyright 2016 The Go Authors. All rights reserved.
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
func (c *sigctxt) regs() *sigcontext { return &(*ucontext)(c.ctxt).uc_mcontext }

func (c *sigctxt) ra() uint64  { return c.regs().sc_regs.ra }
func (c *sigctxt) sp() uint64  { return c.regs().sc_regs.sp }
func (c *sigctxt) gp() uint64  { return c.regs().sc_regs.gp }
func (c *sigctxt) tp() uint64  { return c.regs().sc_regs.tp }
func (c *sigctxt) t0() uint64  { return c.regs().sc_regs.t0 }
func (c *sigctxt) t1() uint64  { return c.regs().sc_regs.t1 }
func (c *sigctxt) t2() uint64  { return c.regs().sc_regs.t2 }
func (c *sigctxt) s0() uint64  { return c.regs().sc_regs.s0 }
func (c *sigctxt) s1() uint64  { return c.regs().sc_regs.s1 }
func (c *sigctxt) a0() uint64  { return c.regs().sc_regs.a0 }
func (c *sigctxt) a1() uint64  { return c.regs().sc_regs.a1 }
func (c *sigctxt) a2() uint64  { return c.regs().sc_regs.a2 }
func (c *sigctxt) a3() uint64  { return c.regs().sc_regs.a3 }
func (c *sigctxt) a4() uint64  { return c.regs().sc_regs.a4 }
func (c *sigctxt) a5() uint64  { return c.regs().sc_regs.a5 }
func (c *sigctxt) a6() uint64  { return c.regs().sc_regs.a6 }
func (c *sigctxt) a7() uint64  { return c.regs().sc_regs.a7 }
func (c *sigctxt) s2() uint64  { return c.regs().sc_regs.s2 }
func (c *sigctxt) s3() uint64  { return c.regs().sc_regs.s3 }
func (c *sigctxt) s4() uint64  { return c.regs().sc_regs.s4 }
func (c *sigctxt) s5() uint64  { return c.regs().sc_regs.s5 }
func (c *sigctxt) s6() uint64  { return c.regs().sc_regs.s6 }
func (c *sigctxt) s7() uint64  { return c.regs().sc_regs.s7 }
func (c *sigctxt) s8() uint64  { return c.regs().sc_regs.s8 }
func (c *sigctxt) s9() uint64  { return c.regs().sc_regs.s9 }
func (c *sigctxt) s10() uint64 { return c.regs().sc_regs.s10 }
func (c *sigctxt) s11() uint64 { return c.regs().sc_regs.s11 }
func (c *sigctxt) t3() uint64  { return c.regs().sc_regs.t3 }
func (c *sigctxt) t4() uint64  { return c.regs().sc_regs.t4 }
func (c *sigctxt) t5() uint64  { return c.regs().sc_regs.t5 }
func (c *sigctxt) t6() uint64  { return c.regs().sc_regs.t6 }

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) pc() uint64 { return c.regs().sc_regs.pc }

func (c *sigctxt) sigcode() uint32 { return uint32(c.info.si_code) }
func (c *sigctxt) sigaddr() uint64 { return c.info.si_addr }

func (c *sigctxt) set_pc(x uint64) { c.regs().sc_regs.pc = x }
func (c *sigctxt) set_ra(x uint64) { c.regs().sc_regs.ra = x }
func (c *sigctxt) set_sp(x uint64) { c.regs().sc_regs.sp = x }
func (c *sigctxt) set_gp(x uint64) { c.regs().sc_regs.gp = x }

func (c *sigctxt) set_sigcode(x uint32) { c.info.si_code = int32(x) }
func (c *sigctxt) set_sigaddr(x uint64) {
	*(*uintptr)(add(unsafe.Pointer(c.info), 2*goarch.PtrSize)) = uintptr(x)
}

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

func (c *sigctxt) ra() uint64  { return uint64(c.regs().sc_ra) }
func (c *sigctxt) sp() uint64  { return uint64(c.regs().sc_sp) }
func (c *sigctxt) gp() uint64  { return uint64(c.regs().sc_gp) }
func (c *sigctxt) tp() uint64  { return uint64(c.regs().sc_tp) }
func (c *sigctxt) t0() uint64  { return uint64(c.regs().sc_t[0]) }
func (c *sigctxt) t1() uint64  { return uint64(c.regs().sc_t[1]) }
func (c *sigctxt) t2() uint64  { return uint64(c.regs().sc_t[2]) }
func (c *sigctxt) s0() uint64  { return uint64(c.regs().sc_s[0]) }
func (c *sigctxt) s1() uint64  { return uint64(c.regs().sc_s[1]) }
func (c *sigctxt) a0() uint64  { return uint64(c.regs().sc_a[0]) }
func (c *sigctxt) a1() uint64  { return uint64(c.regs().sc_a[1]) }
func (c *sigctxt) a2() uint64  { return uint64(c.regs().sc_a[2]) }
func (c *sigctxt) a3() uint64  { return uint64(c.regs().sc_a[3]) }
func (c *sigctxt) a4() uint64  { return uint64(c.regs().sc_a[4]) }
func (c *sigctxt) a5() uint64  { return uint64(c.regs().sc_a[5]) }
func (c *sigctxt) a6() uint64  { return uint64(c.regs().sc_a[6]) }
func (c *sigctxt) a7() uint64  { return uint64(c.regs().sc_a[7]) }
func (c *sigctxt) s2() uint64  { return uint64(c.regs().sc_s[2]) }
func (c *sigctxt) s3() uint64  { return uint64(c.regs().sc_s[3]) }
func (c *sigctxt) s4() uint64  { return uint64(c.regs().sc_s[4]) }
func (c *sigctxt) s5() uint64  { return uint64(c.regs().sc_s[5]) }
func (c *sigctxt) s6() uint64  { return uint64(c.regs().sc_s[6]) }
func (c *sigctxt) s7() uint64  { return uint64(c.regs().sc_s[7]) }
func (c *sigctxt) s8() uint64  { return uint64(c.regs().sc_s[8]) }
func (c *sigctxt) s9() uint64  { return uint64(c.regs().sc_s[9]) }
func (c *sigctxt) s10() uint64 { return uint64(c.regs().sc_s[10]) }
func (c *sigctxt) s11() uint64 { return uint64(c.regs().sc_s[11]) }
func (c *sigctxt) t3() uint64  { return uint64(c.regs().sc_t[3]) }
func (c *sigctxt) t4() uint64  { return uint64(c.regs().sc_t[4]) }
func (c *sigctxt) t5() uint64  { return uint64(c.regs().sc_t[5]) }
func (c *sigctxt) t6() uint64  { return uint64(c.regs().sc_t[6]) }

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) pc() uint64 { return uint64(c.regs().sc_sepc) }

func (c *sigctxt) sigcode() uint64 { return uint64(c.info.si_code) }
func (c *sigctxt) sigaddr() uint64 {
	return *(*uint64)(add(unsafe.Pointer(c.info), 2*goarch.PtrSize))
}

func (c *sigctxt) set_pc(x uint64) { c.regs().sc_sepc = uintptr(x) }
func (c *sigctxt) set_ra(x uint64) { c.regs().sc_ra = uintptr(x) }
func (c *sigctxt) set_sp(x uint64) { c.regs().sc_sp = uintptr(x) }
func (c *sigctxt) set_gp(x uint64) { c.regs().sc_gp = uintptr(x) }

func (c *sigctxt) set_sigcode(x uint32) { c.info.si_code = int32(x) }
func (c *sigctxt) set_sigaddr(x uint64) {
	*(*uintptr)(add(unsafe.Pointer(c.info), 2*goarch.PtrSize)) = uintptr(x)
}

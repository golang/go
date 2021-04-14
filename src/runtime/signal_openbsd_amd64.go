// Copyright 2013 The Go Authors. All rights reserved.
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

func (c *sigctxt) rax() uint64 { return c.regs().sc_rax }
func (c *sigctxt) rbx() uint64 { return c.regs().sc_rbx }
func (c *sigctxt) rcx() uint64 { return c.regs().sc_rcx }
func (c *sigctxt) rdx() uint64 { return c.regs().sc_rdx }
func (c *sigctxt) rdi() uint64 { return c.regs().sc_rdi }
func (c *sigctxt) rsi() uint64 { return c.regs().sc_rsi }
func (c *sigctxt) rbp() uint64 { return c.regs().sc_rbp }
func (c *sigctxt) rsp() uint64 { return c.regs().sc_rsp }
func (c *sigctxt) r8() uint64  { return c.regs().sc_r8 }
func (c *sigctxt) r9() uint64  { return c.regs().sc_r9 }
func (c *sigctxt) r10() uint64 { return c.regs().sc_r10 }
func (c *sigctxt) r11() uint64 { return c.regs().sc_r11 }
func (c *sigctxt) r12() uint64 { return c.regs().sc_r12 }
func (c *sigctxt) r13() uint64 { return c.regs().sc_r13 }
func (c *sigctxt) r14() uint64 { return c.regs().sc_r14 }
func (c *sigctxt) r15() uint64 { return c.regs().sc_r15 }

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) rip() uint64 { return c.regs().sc_rip }

func (c *sigctxt) rflags() uint64  { return c.regs().sc_rflags }
func (c *sigctxt) cs() uint64      { return c.regs().sc_cs }
func (c *sigctxt) fs() uint64      { return c.regs().sc_fs }
func (c *sigctxt) gs() uint64      { return c.regs().sc_gs }
func (c *sigctxt) sigcode() uint64 { return uint64(c.info.si_code) }
func (c *sigctxt) sigaddr() uint64 {
	return *(*uint64)(add(unsafe.Pointer(c.info), 16))
}

func (c *sigctxt) set_rip(x uint64)     { c.regs().sc_rip = x }
func (c *sigctxt) set_rsp(x uint64)     { c.regs().sc_rsp = x }
func (c *sigctxt) set_sigcode(x uint64) { c.info.si_code = int32(x) }
func (c *sigctxt) set_sigaddr(x uint64) {
	*(*uint64)(add(unsafe.Pointer(c.info), 16)) = x
}

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

func (c *sigctxt) eax() uint32 { return c.regs().sc_eax }
func (c *sigctxt) ebx() uint32 { return c.regs().sc_ebx }
func (c *sigctxt) ecx() uint32 { return c.regs().sc_ecx }
func (c *sigctxt) edx() uint32 { return c.regs().sc_edx }
func (c *sigctxt) edi() uint32 { return c.regs().sc_edi }
func (c *sigctxt) esi() uint32 { return c.regs().sc_esi }
func (c *sigctxt) ebp() uint32 { return c.regs().sc_ebp }
func (c *sigctxt) esp() uint32 { return c.regs().sc_esp }

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) eip() uint32 { return c.regs().sc_eip }

func (c *sigctxt) eflags() uint32  { return c.regs().sc_eflags }
func (c *sigctxt) cs() uint32      { return c.regs().sc_cs }
func (c *sigctxt) fs() uint32      { return c.regs().sc_fs }
func (c *sigctxt) gs() uint32      { return c.regs().sc_gs }
func (c *sigctxt) sigcode() uint32 { return uint32(c.info.si_code) }
func (c *sigctxt) sigaddr() uint32 {
	return *(*uint32)(add(unsafe.Pointer(c.info), 12))
}

func (c *sigctxt) set_eip(x uint32)     { c.regs().sc_eip = x }
func (c *sigctxt) set_esp(x uint32)     { c.regs().sc_esp = x }
func (c *sigctxt) set_sigcode(x uint32) { c.info.si_code = int32(x) }
func (c *sigctxt) set_sigaddr(x uint32) {
	*(*uint32)(add(unsafe.Pointer(c.info), 12)) = x
}

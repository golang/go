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
func (c *sigctxt) regs() *mcontextt { return &(*ucontextt)(c.ctxt).uc_mcontext }

func (c *sigctxt) eax() uint32 { return c.regs().__gregs[_REG_EAX] }
func (c *sigctxt) ebx() uint32 { return c.regs().__gregs[_REG_EBX] }
func (c *sigctxt) ecx() uint32 { return c.regs().__gregs[_REG_ECX] }
func (c *sigctxt) edx() uint32 { return c.regs().__gregs[_REG_EDX] }
func (c *sigctxt) edi() uint32 { return c.regs().__gregs[_REG_EDI] }
func (c *sigctxt) esi() uint32 { return c.regs().__gregs[_REG_ESI] }
func (c *sigctxt) ebp() uint32 { return c.regs().__gregs[_REG_EBP] }
func (c *sigctxt) esp() uint32 { return c.regs().__gregs[_REG_UESP] }

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) eip() uint32 { return c.regs().__gregs[_REG_EIP] }

func (c *sigctxt) eflags() uint32  { return c.regs().__gregs[_REG_EFL] }
func (c *sigctxt) cs() uint32      { return c.regs().__gregs[_REG_CS] }
func (c *sigctxt) fs() uint32      { return c.regs().__gregs[_REG_FS] }
func (c *sigctxt) gs() uint32      { return c.regs().__gregs[_REG_GS] }
func (c *sigctxt) sigcode() uint32 { return uint32(c.info._code) }
func (c *sigctxt) sigaddr() uint32 {
	return *(*uint32)(unsafe.Pointer(&c.info._reason[0]))
}

func (c *sigctxt) set_eip(x uint32)     { c.regs().__gregs[_REG_EIP] = x }
func (c *sigctxt) set_esp(x uint32)     { c.regs().__gregs[_REG_UESP] = x }
func (c *sigctxt) set_sigcode(x uint32) { c.info._code = int32(x) }
func (c *sigctxt) set_sigaddr(x uint32) {
	*(*uint32)(unsafe.Pointer(&c.info._reason[0])) = x
}

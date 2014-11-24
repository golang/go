// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

type sigctxt struct {
	info *siginfo
	ctxt unsafe.Pointer
}

func (c *sigctxt) regs() *mcontextt {
	return (*mcontextt)(unsafe.Pointer(&(*ucontextt)(c.ctxt).uc_mcontext))
}
func (c *sigctxt) rax() uint64     { return c.regs().__gregs[_REG_RAX] }
func (c *sigctxt) rbx() uint64     { return c.regs().__gregs[_REG_RBX] }
func (c *sigctxt) rcx() uint64     { return c.regs().__gregs[_REG_RCX] }
func (c *sigctxt) rdx() uint64     { return c.regs().__gregs[_REG_RDX] }
func (c *sigctxt) rdi() uint64     { return c.regs().__gregs[_REG_RDI] }
func (c *sigctxt) rsi() uint64     { return c.regs().__gregs[_REG_RSI] }
func (c *sigctxt) rbp() uint64     { return c.regs().__gregs[_REG_RBP] }
func (c *sigctxt) rsp() uint64     { return c.regs().__gregs[_REG_RSP] }
func (c *sigctxt) r8() uint64      { return c.regs().__gregs[_REG_R8] }
func (c *sigctxt) r9() uint64      { return c.regs().__gregs[_REG_R8] }
func (c *sigctxt) r10() uint64     { return c.regs().__gregs[_REG_R10] }
func (c *sigctxt) r11() uint64     { return c.regs().__gregs[_REG_R11] }
func (c *sigctxt) r12() uint64     { return c.regs().__gregs[_REG_R12] }
func (c *sigctxt) r13() uint64     { return c.regs().__gregs[_REG_R13] }
func (c *sigctxt) r14() uint64     { return c.regs().__gregs[_REG_R14] }
func (c *sigctxt) r15() uint64     { return c.regs().__gregs[_REG_R15] }
func (c *sigctxt) rip() uint64     { return c.regs().__gregs[_REG_RIP] }
func (c *sigctxt) rflags() uint64  { return c.regs().__gregs[_REG_RFLAGS] }
func (c *sigctxt) cs() uint64      { return c.regs().__gregs[_REG_CS] }
func (c *sigctxt) fs() uint64      { return c.regs().__gregs[_REG_FS] }
func (c *sigctxt) gs() uint64      { return c.regs().__gregs[_REG_GS] }
func (c *sigctxt) sigcode() uint64 { return uint64(c.info._code) }
func (c *sigctxt) sigaddr() uint64 {
	return uint64(*(*uint64)(unsafe.Pointer(&c.info._reason[0])))
}

func (c *sigctxt) set_rip(x uint64)     { c.regs().__gregs[_REG_RIP] = x }
func (c *sigctxt) set_rsp(x uint64)     { c.regs().__gregs[_REG_RSP] = x }
func (c *sigctxt) set_sigcode(x uint64) { c.info._code = int32(x) }
func (c *sigctxt) set_sigaddr(x uint64) {
	*(*uint64)(unsafe.Pointer(&c.info._reason[0])) = x
}

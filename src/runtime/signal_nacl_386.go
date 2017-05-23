// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

type sigctxt struct {
	info *siginfo
	ctxt unsafe.Pointer
}

func (c *sigctxt) regs() *excregs386 { return &(*exccontext)(c.ctxt).regs }
func (c *sigctxt) eax() uint32       { return c.regs().eax }
func (c *sigctxt) ebx() uint32       { return c.regs().ebx }
func (c *sigctxt) ecx() uint32       { return c.regs().ecx }
func (c *sigctxt) edx() uint32       { return c.regs().edx }
func (c *sigctxt) edi() uint32       { return c.regs().edi }
func (c *sigctxt) esi() uint32       { return c.regs().esi }
func (c *sigctxt) ebp() uint32       { return c.regs().ebp }
func (c *sigctxt) esp() uint32       { return c.regs().esp }
func (c *sigctxt) eip() uint32       { return c.regs().eip }
func (c *sigctxt) eflags() uint32    { return c.regs().eflags }
func (c *sigctxt) cs() uint32        { return ^uint32(0) }
func (c *sigctxt) fs() uint32        { return ^uint32(0) }
func (c *sigctxt) gs() uint32        { return ^uint32(0) }
func (c *sigctxt) sigcode() uint32   { return ^uint32(0) }
func (c *sigctxt) sigaddr() uint32   { return 0 }

func (c *sigctxt) set_eip(x uint32)     { c.regs().eip = x }
func (c *sigctxt) set_esp(x uint32)     { c.regs().esp = x }
func (c *sigctxt) set_sigcode(x uint32) {}
func (c *sigctxt) set_sigaddr(x uint32) {}

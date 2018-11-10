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
func (c *sigctxt) regs() *regs32 { return &(*ucontext)(c.ctxt).uc_mcontext.ss }

func (c *sigctxt) eax() uint32 { return c.regs().eax }
func (c *sigctxt) ebx() uint32 { return c.regs().ebx }
func (c *sigctxt) ecx() uint32 { return c.regs().ecx }
func (c *sigctxt) edx() uint32 { return c.regs().edx }
func (c *sigctxt) edi() uint32 { return c.regs().edi }
func (c *sigctxt) esi() uint32 { return c.regs().esi }
func (c *sigctxt) ebp() uint32 { return c.regs().ebp }
func (c *sigctxt) esp() uint32 { return c.regs().esp }

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) eip() uint32 { return c.regs().eip }

func (c *sigctxt) eflags() uint32  { return c.regs().eflags }
func (c *sigctxt) cs() uint32      { return c.regs().cs }
func (c *sigctxt) fs() uint32      { return c.regs().fs }
func (c *sigctxt) gs() uint32      { return c.regs().gs }
func (c *sigctxt) sigcode() uint32 { return uint32(c.info.si_code) }
func (c *sigctxt) sigaddr() uint32 { return c.info.si_addr }

func (c *sigctxt) set_eip(x uint32)     { c.regs().eip = x }
func (c *sigctxt) set_esp(x uint32)     { c.regs().esp = x }
func (c *sigctxt) set_sigcode(x uint32) { c.info.si_code = int32(x) }
func (c *sigctxt) set_sigaddr(x uint32) { c.info.si_addr = x }

func (c *sigctxt) fixsigcode(sig uint32) {
	switch sig {
	case _SIGTRAP:
		// OS X sets c.sigcode() == TRAP_BRKPT unconditionally for all SIGTRAPs,
		// leaving no way to distinguish a breakpoint-induced SIGTRAP
		// from an asynchronous signal SIGTRAP.
		// They all look breakpoint-induced by default.
		// Try looking at the code to see if it's a breakpoint.
		// The assumption is that we're very unlikely to get an
		// asynchronous SIGTRAP at just the moment that the
		// PC started to point at unmapped memory.
		pc := uintptr(c.eip())
		// OS X will leave the pc just after the INT 3 instruction.
		// INT 3 is usually 1 byte, but there is a 2-byte form.
		code := (*[2]byte)(unsafe.Pointer(pc - 2))
		if code[1] != 0xCC && (code[0] != 0xCD || code[1] != 3) {
			// SIGTRAP on something other than INT 3.
			c.set_sigcode(_SI_USER)
		}
	}
}

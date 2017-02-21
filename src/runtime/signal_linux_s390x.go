// Copyright 2016 The Go Authors. All rights reserved.
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
func (c *sigctxt) regs() *sigcontext {
	return (*sigcontext)(unsafe.Pointer(&(*ucontext)(c.ctxt).uc_mcontext))
}

func (c *sigctxt) r0() uint64   { return c.regs().gregs[0] }
func (c *sigctxt) r1() uint64   { return c.regs().gregs[1] }
func (c *sigctxt) r2() uint64   { return c.regs().gregs[2] }
func (c *sigctxt) r3() uint64   { return c.regs().gregs[3] }
func (c *sigctxt) r4() uint64   { return c.regs().gregs[4] }
func (c *sigctxt) r5() uint64   { return c.regs().gregs[5] }
func (c *sigctxt) r6() uint64   { return c.regs().gregs[6] }
func (c *sigctxt) r7() uint64   { return c.regs().gregs[7] }
func (c *sigctxt) r8() uint64   { return c.regs().gregs[8] }
func (c *sigctxt) r9() uint64   { return c.regs().gregs[9] }
func (c *sigctxt) r10() uint64  { return c.regs().gregs[10] }
func (c *sigctxt) r11() uint64  { return c.regs().gregs[11] }
func (c *sigctxt) r12() uint64  { return c.regs().gregs[12] }
func (c *sigctxt) r13() uint64  { return c.regs().gregs[13] }
func (c *sigctxt) r14() uint64  { return c.regs().gregs[14] }
func (c *sigctxt) r15() uint64  { return c.regs().gregs[15] }
func (c *sigctxt) link() uint64 { return c.regs().gregs[14] }
func (c *sigctxt) sp() uint64   { return c.regs().gregs[15] }

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) pc() uint64 { return c.regs().psw_addr }

func (c *sigctxt) sigcode() uint32 { return uint32(c.info.si_code) }
func (c *sigctxt) sigaddr() uint64 { return c.info.si_addr }

func (c *sigctxt) set_r0(x uint64)      { c.regs().gregs[0] = x }
func (c *sigctxt) set_r13(x uint64)     { c.regs().gregs[13] = x }
func (c *sigctxt) set_link(x uint64)    { c.regs().gregs[14] = x }
func (c *sigctxt) set_sp(x uint64)      { c.regs().gregs[15] = x }
func (c *sigctxt) set_pc(x uint64)      { c.regs().psw_addr = x }
func (c *sigctxt) set_sigcode(x uint32) { c.info.si_code = int32(x) }
func (c *sigctxt) set_sigaddr(x uint64) {
	*(*uintptr)(add(unsafe.Pointer(c.info), 2*sys.PtrSize)) = uintptr(x)
}

func dumpregs(c *sigctxt) {
	print("r0   ", hex(c.r0()), "\t")
	print("r1   ", hex(c.r1()), "\n")
	print("r2   ", hex(c.r2()), "\t")
	print("r3   ", hex(c.r3()), "\n")
	print("r4   ", hex(c.r4()), "\t")
	print("r5   ", hex(c.r5()), "\n")
	print("r6   ", hex(c.r6()), "\t")
	print("r7   ", hex(c.r7()), "\n")
	print("r8   ", hex(c.r8()), "\t")
	print("r9   ", hex(c.r9()), "\n")
	print("r10  ", hex(c.r10()), "\t")
	print("r11  ", hex(c.r11()), "\n")
	print("r12  ", hex(c.r12()), "\t")
	print("r13  ", hex(c.r13()), "\n")
	print("r14  ", hex(c.r14()), "\t")
	print("r15  ", hex(c.r15()), "\n")
	print("pc   ", hex(c.pc()), "\t")
	print("link ", hex(c.link()), "\n")
}

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) sigpc() uintptr { return uintptr(c.pc()) }

func (c *sigctxt) sigsp() uintptr { return uintptr(c.sp()) }
func (c *sigctxt) siglr() uintptr { return uintptr(c.link()) }
func (c *sigctxt) fault() uintptr { return uintptr(c.sigaddr()) }

// preparePanic sets up the stack to look like a call to sigpanic.
func (c *sigctxt) preparePanic(sig uint32, gp *g) {
	// We arrange link, and pc to pretend the panicking
	// function calls sigpanic directly.
	// Always save LINK to stack so that panics in leaf
	// functions are correctly handled. This smashes
	// the stack frame but we're not going back there
	// anyway.
	sp := c.sp() - sys.MinFrameSize
	c.set_sp(sp)
	*(*uint64)(unsafe.Pointer(uintptr(sp))) = c.link()

	pc := uintptr(gp.sigpc)

	// If we don't recognize the PC as code
	// but we do recognize the link register as code,
	// then assume this was a call to non-code and treat like
	// pc == 0, to make unwinding show the context.
	if pc != 0 && !findfunc(pc).valid() && findfunc(uintptr(c.link())).valid() {
		pc = 0
	}

	// Don't bother saving PC if it's zero, which is
	// probably a call to a nil func: the old link register
	// is more useful in the stack trace.
	if pc != 0 {
		c.set_link(uint64(pc))
	}

	// In case we are panicking from external C code
	c.set_r0(0)
	c.set_r13(uint64(uintptr(unsafe.Pointer(gp))))
	c.set_pc(uint64(funcPC(sigpanic)))
}

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// NOTE(rsc): _CONTEXT_CONTROL is actually 0x400001 and should include PC, SP, and LR.
// However, empirically, LR doesn't come along on Windows 10
// unless you also set _CONTEXT_INTEGER (0x400002).
// Without LR, we skip over the next-to-bottom function in profiles
// when the bottom function is frameless.
// So we set both here, to make a working _CONTEXT_CONTROL.
const _CONTEXT_CONTROL = 0x400003

type neon128 struct {
	low  uint64
	high int64
}

// See https://docs.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-arm64_nt_context
type context struct {
	contextflags uint32
	cpsr         uint32
	x            [31]uint64 // fp is x[29], lr is x[30]
	xsp          uint64
	pc           uint64
	v            [32]neon128
	fpcr         uint32
	fpsr         uint32
	bcr          [8]uint32
	bvr          [8]uint64
	wcr          [2]uint32
	wvr          [2]uint64
}

func (c *context) ip() uintptr { return uintptr(c.pc) }
func (c *context) sp() uintptr { return uintptr(c.xsp) }
func (c *context) lr() uintptr { return uintptr(c.x[30]) }

func (c *context) set_ip(x uintptr) { c.pc = uint64(x) }
func (c *context) set_sp(x uintptr) { c.xsp = uint64(x) }
func (c *context) set_lr(x uintptr) { c.x[30] = uint64(x) }

func dumpregs(r *context) {
	print("r0   ", hex(r.x[0]), "\n")
	print("r1   ", hex(r.x[1]), "\n")
	print("r2   ", hex(r.x[2]), "\n")
	print("r3   ", hex(r.x[3]), "\n")
	print("r4   ", hex(r.x[4]), "\n")
	print("r5   ", hex(r.x[5]), "\n")
	print("r6   ", hex(r.x[6]), "\n")
	print("r7   ", hex(r.x[7]), "\n")
	print("r8   ", hex(r.x[8]), "\n")
	print("r9   ", hex(r.x[9]), "\n")
	print("r10  ", hex(r.x[10]), "\n")
	print("r11  ", hex(r.x[11]), "\n")
	print("r12  ", hex(r.x[12]), "\n")
	print("r13  ", hex(r.x[13]), "\n")
	print("r14  ", hex(r.x[14]), "\n")
	print("r15  ", hex(r.x[15]), "\n")
	print("r16  ", hex(r.x[16]), "\n")
	print("r17  ", hex(r.x[17]), "\n")
	print("r18  ", hex(r.x[18]), "\n")
	print("r19  ", hex(r.x[19]), "\n")
	print("r20  ", hex(r.x[20]), "\n")
	print("r21  ", hex(r.x[21]), "\n")
	print("r22  ", hex(r.x[22]), "\n")
	print("r23  ", hex(r.x[23]), "\n")
	print("r24  ", hex(r.x[24]), "\n")
	print("r25  ", hex(r.x[25]), "\n")
	print("r26  ", hex(r.x[26]), "\n")
	print("r27  ", hex(r.x[27]), "\n")
	print("r28  ", hex(r.x[28]), "\n")
	print("r29  ", hex(r.x[29]), "\n")
	print("lr   ", hex(r.x[30]), "\n")
	print("sp   ", hex(r.xsp), "\n")
	print("pc   ", hex(r.pc), "\n")
	print("cpsr ", hex(r.cpsr), "\n")
}

func stackcheck() {
	// TODO: not implemented on ARM
}

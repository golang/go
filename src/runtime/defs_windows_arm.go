// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/goarch"
	"unsafe"
)

// NOTE(rsc): _CONTEXT_CONTROL is actually 0x200001 and should include PC, SP, and LR.
// However, empirically, LR doesn't come along on Windows 10
// unless you also set _CONTEXT_INTEGER (0x200002).
// Without LR, we skip over the next-to-bottom function in profiles
// when the bottom function is frameless.
// So we set both here, to make a working _CONTEXT_CONTROL.
const _CONTEXT_CONTROL = 0x200003

type neon128 struct {
	low  uint64
	high int64
}

type context struct {
	contextflags uint32
	r0           uint32
	r1           uint32
	r2           uint32
	r3           uint32
	r4           uint32
	r5           uint32
	r6           uint32
	r7           uint32
	r8           uint32
	r9           uint32
	r10          uint32
	r11          uint32
	r12          uint32

	spr  uint32
	lrr  uint32
	pc   uint32
	cpsr uint32

	fpscr   uint32
	padding uint32

	floatNeon [16]neon128

	bvr      [8]uint32
	bcr      [8]uint32
	wvr      [1]uint32
	wcr      [1]uint32
	padding2 [2]uint32
}

//go:nosplit
func (c *context) ip() uintptr { return uintptr(c.pc) }
//go:nosplit
func (c *context) sp() uintptr { return uintptr(c.spr) }
//go:nosplit
func (c *context) lr() uintptr { return uintptr(c.lrr) }

//go:nosplit
func (c *context) set_ip(x uintptr) { c.pc = uint32(x) }
//go:nosplit
func (c *context) set_sp(x uintptr) { c.spr = uint32(x) }
//go:nosplit
func (c *context) set_lr(x uintptr) { c.lrr = uint32(x) }

// arm does not have frame pointer register.
func (c *context) set_fp(x uintptr) {}

func (c *context) pushCall(targetPC, resumePC uintptr) {
	// Push LR. The injected call is responsible
	// for restoring LR. gentraceback is aware of
	// this extra slot. See sigctxt.pushCall in
	// signal_arm.go.
	sp := c.sp() - goarch.StackAlign
	c.set_sp(sp)
	*(*uint32)(unsafe.Pointer(sp)) = uint32(c.lr())
	c.set_lr(resumePC)
	c.set_ip(targetPC)
}

func prepareContextForSigResume(c *context) {
	c.r0 = c.spr
	c.r1 = c.pc
}

func dumpregs(r *context) {
	print("r0   ", hex(r.r0), "\n")
	print("r1   ", hex(r.r1), "\n")
	print("r2   ", hex(r.r2), "\n")
	print("r3   ", hex(r.r3), "\n")
	print("r4   ", hex(r.r4), "\n")
	print("r5   ", hex(r.r5), "\n")
	print("r6   ", hex(r.r6), "\n")
	print("r7   ", hex(r.r7), "\n")
	print("r8   ", hex(r.r8), "\n")
	print("r9   ", hex(r.r9), "\n")
	print("r10  ", hex(r.r10), "\n")
	print("r11  ", hex(r.r11), "\n")
	print("r12  ", hex(r.r12), "\n")
	print("sp   ", hex(r.spr), "\n")
	print("lr   ", hex(r.lrr), "\n")
	print("pc   ", hex(r.pc), "\n")
	print("cpsr ", hex(r.cpsr), "\n")
}

func stackcheck() {
	// TODO: not implemented on ARM
}

type _DISPATCHER_CONTEXT struct {
	controlPc        uint32
	imageBase        uint32
	functionEntry    uintptr
	establisherFrame uint32
	targetIp         uint32
	context          *context
	languageHandler  uintptr
	handlerData      uintptr
}

func (c *_DISPATCHER_CONTEXT) ctx() *context {
	return c.context
}

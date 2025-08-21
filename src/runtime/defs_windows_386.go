// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/goarch"
	"unsafe"
)

const _CONTEXT_CONTROL = 0x10001

type floatingsavearea struct {
	controlword   uint32
	statusword    uint32
	tagword       uint32
	erroroffset   uint32
	errorselector uint32
	dataoffset    uint32
	dataselector  uint32
	registerarea  [80]uint8
	cr0npxstate   uint32
}

type context struct {
	contextflags      uint32
	dr0               uint32
	dr1               uint32
	dr2               uint32
	dr3               uint32
	dr6               uint32
	dr7               uint32
	floatsave         floatingsavearea
	seggs             uint32
	segfs             uint32
	seges             uint32
	segds             uint32
	edi               uint32
	esi               uint32
	ebx               uint32
	edx               uint32
	ecx               uint32
	eax               uint32
	ebp               uint32
	eip               uint32
	segcs             uint32
	eflags            uint32
	esp               uint32
	segss             uint32
	extendedregisters [512]uint8
}

//go:nosplit
func (c *context) ip() uintptr { return uintptr(c.eip) }
//go:nosplit
func (c *context) sp() uintptr { return uintptr(c.esp) }

// 386 does not have link register, so this returns 0.
//go:nosplit
func (c *context) lr() uintptr      { return 0 }
//go:nosplit
func (c *context) set_lr(x uintptr) {}

//go:nosplit
func (c *context) set_ip(x uintptr) { c.eip = uint32(x) }
//go:nosplit
func (c *context) set_sp(x uintptr) { c.esp = uint32(x) }

// 386 does not have frame pointer register.
func (c *context) set_fp(x uintptr) {}

func (c *context) pushCall(targetPC, resumePC uintptr) {
	sp := c.sp() - goarch.StackAlign
	*(*uintptr)(unsafe.Pointer(sp)) = resumePC
	c.set_sp(sp)
	c.set_ip(targetPC)
}

func prepareContextForSigResume(c *context) {
	c.edx = c.esp
	c.ecx = c.eip
}

func dumpregs(r *context) {
	print("eax     ", hex(r.eax), "\n")
	print("ebx     ", hex(r.ebx), "\n")
	print("ecx     ", hex(r.ecx), "\n")
	print("edx     ", hex(r.edx), "\n")
	print("edi     ", hex(r.edi), "\n")
	print("esi     ", hex(r.esi), "\n")
	print("ebp     ", hex(r.ebp), "\n")
	print("esp     ", hex(r.esp), "\n")
	print("eip     ", hex(r.eip), "\n")
	print("eflags  ", hex(r.eflags), "\n")
	print("cs      ", hex(r.segcs), "\n")
	print("fs      ", hex(r.segfs), "\n")
	print("gs      ", hex(r.seggs), "\n")
}

// _DISPATCHER_CONTEXT is not defined on 386.
type _DISPATCHER_CONTEXT struct{}

func (c *_DISPATCHER_CONTEXT) ctx() *context {
	return nil
}

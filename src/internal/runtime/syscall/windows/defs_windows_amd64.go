// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package windows

import (
	"internal/goarch"
	"unsafe"
)

const CONTEXT_CONTROL = 0x100001

type M128 struct {
	Low  uint64
	High int64
}

type Context struct {
	P1Home               uint64
	P2Home               uint64
	P3Home               uint64
	P4Home               uint64
	P5Home               uint64
	P6Home               uint64
	ContextFlags         uint32
	MxCsr                uint32
	SegCs                uint16
	SegDs                uint16
	SegEs                uint16
	SegFs                uint16
	SegGs                uint16
	SegSs                uint16
	EFlags               uint32
	DR0                  uint64
	DR1                  uint64
	DR2                  uint64
	DR3                  uint64
	DR6                  uint64
	DR7                  uint64
	Rax                  uint64
	Rcx                  uint64
	Rdx                  uint64
	Rbx                  uint64
	Rsp                  uint64
	Rbp                  uint64
	Rsi                  uint64
	Rdi                  uint64
	R8                   uint64
	R9                   uint64
	R10                  uint64
	R11                  uint64
	R12                  uint64
	R13                  uint64
	R14                  uint64
	R15                  uint64
	Rip                  uint64
	_                    [512]byte
	VectorRegister       [26]M128
	VectorControl        uint64
	DebugControl         uint64
	LastBranchToRip      uint64
	LastBranchFromRip    uint64
	LastExceptionToRip   uint64
	LastExceptionFromRip uint64
}

func (c *Context) PC() uintptr { return uintptr(c.Rip) }
func (c *Context) SP() uintptr { return uintptr(c.Rsp) }

// AMD64 does not have link register, so this returns 0.
func (c *Context) LR() uintptr     { return 0 }
func (c *Context) SetLR(x uintptr) {}

func (c *Context) SetPC(x uintptr) { c.Rip = uint64(x) }
func (c *Context) SetSP(x uintptr) { c.Rsp = uint64(x) }
func (c *Context) SetFP(x uintptr) { c.Rbp = uint64(x) }

func (c *Context) PushCall(targetPC, resumePC uintptr) {
	sp := c.SP() - goarch.StackAlign
	*(*uintptr)(unsafe.Pointer(sp)) = resumePC
	c.SetSP(sp)
	c.SetPC(targetPC)
}

type DISPATCHER_CONTEXT struct {
	ControlPc        uint64
	ImageBase        uint64
	FunctionEntry    uintptr
	EstablisherFrame uint64
	TargetIp         uint64
	Context          *Context
	LanguageHandler  uintptr
	HandlerData      uintptr
}

func (c *DISPATCHER_CONTEXT) Ctx() *Context {
	return c.Context
}

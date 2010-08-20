// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proc

import (
	"os"
	"strconv"
	"syscall"
)

type _386Regs struct {
	syscall.PtraceRegs
	setter func(*syscall.PtraceRegs) os.Error
}

var names = []string{
	"eax",
	"ebx",
	"ecx",
	"edx",
	"esi",
	"edi",
	"ebp",
	"esp",
	"eip",
	"eflags",
	"cs",
	"ss",
	"ds",
	"es",
	"fs",
	"gs",
}

func (r *_386Regs) PC() Word { return Word(r.Eip) }

func (r *_386Regs) SetPC(val Word) os.Error {
	r.Eip = int32(val)
	return r.setter(&r.PtraceRegs)
}

func (r *_386Regs) Link() Word {
	// TODO(austin)
	panic("No link register")
}

func (r *_386Regs) SetLink(val Word) os.Error { panic("No link register") }

func (r *_386Regs) SP() Word { return Word(r.Esp) }

func (r *_386Regs) SetSP(val Word) os.Error {
	r.Esp = int32(val)
	return r.setter(&r.PtraceRegs)
}

func (r *_386Regs) Names() []string { return names }

func (r *_386Regs) Get(i int) Word {
	switch i {
	case 0:
		return Word(uint32(r.Eax))
	case 1:
		return Word(uint32(r.Ebx))
	case 2:
		return Word(uint32(r.Ecx))
	case 3:
		return Word(uint32(r.Edx))
	case 4:
		return Word(uint32(r.Esi))
	case 5:
		return Word(uint32(r.Edi))
	case 6:
		return Word(uint32(r.Ebp))
	case 7:
		return Word(uint32(r.Esp))
	case 8:
		return Word(uint32(r.Eip))
	case 9:
		return Word(uint32(r.Eflags))
	case 10:
		return Word(r.Xcs)
	case 11:
		return Word(r.Xss)
	case 12:
		return Word(r.Xds)
	case 13:
		return Word(r.Xes)
	case 14:
		return Word(r.Xfs)
	case 15:
		return Word(r.Xgs)
	}
	panic("invalid register index " + strconv.Itoa(i))
}

func (r *_386Regs) Set(i int, val Word) os.Error {
	switch i {
	case 0:
		r.Eax = int32(val)
	case 1:
		r.Ebx = int32(val)
	case 2:
		r.Ecx = int32(val)
	case 3:
		r.Edx = int32(val)
	case 4:
		r.Esi = int32(val)
	case 5:
		r.Edi = int32(val)
	case 6:
		r.Ebp = int32(val)
	case 7:
		r.Esp = int32(val)
	case 8:
		r.Eip = int32(val)
	case 9:
		r.Eflags = int32(val)
	case 10:
		r.Xcs = int32(val)
	case 11:
		r.Xss = int32(val)
	case 12:
		r.Xds = int32(val)
	case 13:
		r.Xes = int32(val)
	case 14:
		r.Xfs = int32(val)
	case 15:
		r.Xgs = int32(val)
	default:
		panic("invalid register index " + strconv.Itoa(i))
	}
	return r.setter(&r.PtraceRegs)
}

func newRegs(regs *syscall.PtraceRegs, setter func(*syscall.PtraceRegs) os.Error) Regs {
	res := _386Regs{}
	res.PtraceRegs = *regs
	res.setter = setter
	return &res
}

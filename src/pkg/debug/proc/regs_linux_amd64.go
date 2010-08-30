// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proc

import (
	"os"
	"strconv"
	"syscall"
)

type amd64Regs struct {
	syscall.PtraceRegs
	setter func(*syscall.PtraceRegs) os.Error
}

var names = [...]string{
	"rax",
	"rbx",
	"rcx",
	"rdx",
	"rsi",
	"rdi",
	"rbp",
	"rsp",
	"r8",
	"r9",
	"r10",
	"r11",
	"r12",
	"r13",
	"r14",
	"r15",
	"rip",
	"eflags",
	"cs",
	"ss",
	"ds",
	"es",
	"fs",
	"gs",

	// PtraceRegs contains these registers, but I don't think
	// they're actually meaningful.
	//"orig_rax",
	//"fs_base",
	//"gs_base",
}

func (r *amd64Regs) PC() Word { return Word(r.Rip) }

func (r *amd64Regs) SetPC(val Word) os.Error {
	r.Rip = uint64(val)
	return r.setter(&r.PtraceRegs)
}

func (r *amd64Regs) Link() Word {
	// TODO(austin)
	panic("No link register")
}

func (r *amd64Regs) SetLink(val Word) os.Error {
	panic("No link register")
}

func (r *amd64Regs) SP() Word { return Word(r.Rsp) }

func (r *amd64Regs) SetSP(val Word) os.Error {
	r.Rsp = uint64(val)
	return r.setter(&r.PtraceRegs)
}

func (r *amd64Regs) Names() []string { return names[0:] }

func (r *amd64Regs) Get(i int) Word {
	switch i {
	case 0:
		return Word(r.Rax)
	case 1:
		return Word(r.Rbx)
	case 2:
		return Word(r.Rcx)
	case 3:
		return Word(r.Rdx)
	case 4:
		return Word(r.Rsi)
	case 5:
		return Word(r.Rdi)
	case 6:
		return Word(r.Rbp)
	case 7:
		return Word(r.Rsp)
	case 8:
		return Word(r.R8)
	case 9:
		return Word(r.R9)
	case 10:
		return Word(r.R10)
	case 11:
		return Word(r.R11)
	case 12:
		return Word(r.R12)
	case 13:
		return Word(r.R13)
	case 14:
		return Word(r.R14)
	case 15:
		return Word(r.R15)
	case 16:
		return Word(r.Rip)
	case 17:
		return Word(r.Eflags)
	case 18:
		return Word(r.Cs)
	case 19:
		return Word(r.Ss)
	case 20:
		return Word(r.Ds)
	case 21:
		return Word(r.Es)
	case 22:
		return Word(r.Fs)
	case 23:
		return Word(r.Gs)
	}
	panic("invalid register index " + strconv.Itoa(i))
}

func (r *amd64Regs) Set(i int, val Word) os.Error {
	switch i {
	case 0:
		r.Rax = uint64(val)
	case 1:
		r.Rbx = uint64(val)
	case 2:
		r.Rcx = uint64(val)
	case 3:
		r.Rdx = uint64(val)
	case 4:
		r.Rsi = uint64(val)
	case 5:
		r.Rdi = uint64(val)
	case 6:
		r.Rbp = uint64(val)
	case 7:
		r.Rsp = uint64(val)
	case 8:
		r.R8 = uint64(val)
	case 9:
		r.R9 = uint64(val)
	case 10:
		r.R10 = uint64(val)
	case 11:
		r.R11 = uint64(val)
	case 12:
		r.R12 = uint64(val)
	case 13:
		r.R13 = uint64(val)
	case 14:
		r.R14 = uint64(val)
	case 15:
		r.R15 = uint64(val)
	case 16:
		r.Rip = uint64(val)
	case 17:
		r.Eflags = uint64(val)
	case 18:
		r.Cs = uint64(val)
	case 19:
		r.Ss = uint64(val)
	case 20:
		r.Ds = uint64(val)
	case 21:
		r.Es = uint64(val)
	case 22:
		r.Fs = uint64(val)
	case 23:
		r.Gs = uint64(val)
	default:
		panic("invalid register index " + strconv.Itoa(i))
	}
	return r.setter(&r.PtraceRegs)
}

func newRegs(regs *syscall.PtraceRegs, setter func(*syscall.PtraceRegs) os.Error) Regs {
	res := amd64Regs{}
	res.PtraceRegs = *regs
	res.setter = setter
	return &res
}

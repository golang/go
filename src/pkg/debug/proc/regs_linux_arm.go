// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proc

import (
	"os"
	"syscall"
)

// TODO(kaib): add support

type armRegs struct{}

func (r *armRegs) PC() Word { return Word(0) }

func (r *armRegs) SetPC(val Word) os.Error { return nil }

func (r *armRegs) Link() Word { return Word(0) }

func (r *armRegs) SetLink(val Word) os.Error { return nil }

func (r *armRegs) SP() Word { return Word(0) }

func (r *armRegs) SetSP(val Word) os.Error { return nil }

func (r *armRegs) Names() []string { return nil }

func (r *armRegs) Get(i int) Word { return Word(0) }

func (r *armRegs) Set(i int, val Word) os.Error {
	return nil
}

func newRegs(regs *syscall.PtraceRegs, setter func(*syscall.PtraceRegs) os.Error) Regs {
	res := armRegs{}
	return &res
}

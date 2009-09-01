// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proc

import "syscall"

func newRegs(regs *syscall.PtraceRegs, setter func (*syscall.PtraceRegs) os.Error) Regs {
	panic("newRegs unimplemented on darwin/386");
}

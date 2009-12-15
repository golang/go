// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proc

import (
	"os"
	"syscall"
)

// Process tracing is not supported on Native Client.

func Attach(pid int) (Process, os.Error) {
	return nil, os.NewSyscallError("ptrace", syscall.ENACL)
}

func ForkExec(argv0 string, argv []string, envv []string, dir string, fd []*os.File) (Process, os.Error) {
	return nil, os.NewSyscallError("fork/exec", syscall.ENACL)
}

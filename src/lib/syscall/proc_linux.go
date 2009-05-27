// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

// Process operations for Linux
// TODO:
// - getrlimit, setrlimit

import (
	"syscall";
	"unsafe";
)

func Getrusage(who int64, usage *Rusage) (ret, errno int64) {
	r1, r2, err := Syscall(SYS_GETRUSAGE, who, int64(uintptr(unsafe.Pointer(usage))), 0);
	return r1, err
}

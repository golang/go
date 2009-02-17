// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

import (
	"os";
	"syscall";
	"unsafe";
)

func Sleep(ns int64) *os.Error {
	var tv syscall.Timeval;
	syscall.Nstotimeval(ns, &tv);
	r1, r2, err := syscall.Syscall6(syscall.SYS_SELECT, 0, 0, 0, 0,
		int64(uintptr(unsafe.Pointer(&tv))), 0);
	return os.ErrnoToError(err);
}


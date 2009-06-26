// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

import (
	"os";
	"syscall";
	"unsafe";
)

// Sleep pauses the current goroutine for ns nanoseconds.
// It returns os.EINTR if interrupted.
func Sleep(ns int64) os.Error {
	return os.NewSyscallError("sleep", syscall.Sleep(ns));
}

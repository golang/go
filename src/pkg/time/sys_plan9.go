// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

import (
	"os"
	"syscall"
)

func sysSleep(t int64) error {
	err := syscall.Sleep(t)
	if err != nil {
		return os.NewSyscallError("sleep", err)
	}
	return nil
}

// for testing: whatever interrupts a sleep
func interrupt() {
	// cannot predict pid, don't want to kill group
}

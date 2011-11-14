// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux openbsd

package time

import (
	"os"
	"syscall"
)

// for testing: whatever interrupts a sleep
func interrupt() {
	syscall.Kill(os.Getpid(), syscall.SIGCHLD)
}

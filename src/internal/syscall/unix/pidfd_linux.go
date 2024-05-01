// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

import "syscall"

func PidFDSendSignal(pidfd uintptr, s syscall.Signal) error {
	_, _, errno := syscall.Syscall(pidfdSendSignalTrap, pidfd, uintptr(s), 0)
	if errno != 0 {
		return errno
	}
	return nil
}

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"internal/syscall/windows"
	"syscall"
)

func hostname() (name string, err error) {
	// Use PhysicalDnsHostname to uniquely identify host in a cluster
	const format = windows.ComputerNamePhysicalDnsHostname

	n := uint32(64)
	for {
		b := make([]uint16, n)
		err := windows.GetComputerNameEx(format, &b[0], &n)
		if err == nil {
			return syscall.UTF16ToString(b[:n]), nil
		}
		if err != syscall.ERROR_MORE_DATA {
			return "", NewSyscallError("ComputerNameEx", err)
		}

		// If we received a ERROR_MORE_DATA, but n doesn't get larger,
		// something has gone wrong and we may be in an infinite loop
		if n <= uint32(len(b)) {
			return "", NewSyscallError("ComputerNameEx", err)
		}
	}
}

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nettest

import "syscall"

func supportsRawSocket() bool {
	// From http://msdn.microsoft.com/en-us/library/windows/desktop/ms740548.aspx:
	// Note: To use a socket of type SOCK_RAW requires administrative privileges.
	// Users running Winsock applications that use raw sockets must be a member of
	// the Administrators group on the local computer, otherwise raw socket calls
	// will fail with an error code of WSAEACCES. On Windows Vista and later, access
	// for raw sockets is enforced at socket creation. In earlier versions of Windows,
	// access for raw sockets is enforced during other socket operations.
	for _, af := range []int{syscall.AF_INET, syscall.AF_INET6} {
		s, err := syscall.Socket(af, syscall.SOCK_RAW, 0)
		if err != nil {
			continue
		}
		syscall.Closesocket(s)
		return true
	}
	return false
}

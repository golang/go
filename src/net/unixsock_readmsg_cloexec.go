// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || freebsd || solaris

package net

import "syscall"

const readMsgFlags = 0

func setReadMsgCloseOnExec(oob []byte) {
	scms, err := syscall.ParseSocketControlMessage(oob)
	if err != nil {
		return
	}

	for _, scm := range scms {
		if scm.Header.Level == syscall.SOL_SOCKET && scm.Header.Type == syscall.SCM_RIGHTS {
			fds, err := syscall.ParseUnixRights(&scm)
			if err != nil {
				continue
			}
			for _, fd := range fds {
				syscall.CloseOnExec(fd)
			}
		}
	}
}

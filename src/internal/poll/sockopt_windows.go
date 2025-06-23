// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll

import "syscall"

// WSAIoctl wraps the WSAIoctl network call.
func (fd *FD) WSAIoctl(iocc uint32, inbuf *byte, cbif uint32, outbuf *byte, cbob uint32, cbbr *uint32, overlapped *syscall.Overlapped, completionRoutine uintptr) error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return syscall.WSAIoctl(fd.Sysfd, iocc, inbuf, cbif, outbuf, cbob, cbbr, overlapped, completionRoutine)
}

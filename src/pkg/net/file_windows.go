// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"os"
	"syscall"
)

func FileConn(f *os.File) (c Conn, err os.Error) {
	// TODO: Implement this
	return nil, os.NewSyscallError("FileConn", syscall.EWINDOWS)
}

func FileListener(f *os.File) (l Listener, err os.Error) {
	// TODO: Implement this
	return nil, os.NewSyscallError("FileListener", syscall.EWINDOWS)
}

func FilePacketConn(f *os.File) (c PacketConn, err os.Error) {
	// TODO: Implement this
	return nil, os.NewSyscallError("FilePacketConn", syscall.EWINDOWS)
}

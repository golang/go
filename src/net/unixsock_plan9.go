// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"context"
	"os"
	"syscall"
)

func (c *UnixConn) readFrom(b []byte) (int, *UnixAddr, error) {
	return 0, nil, syscall.EPLAN9
}

func (c *UnixConn) readMsg(b, oob []byte) (n, oobn, flags int, addr *UnixAddr, err error) {
	return 0, 0, 0, nil, syscall.EPLAN9
}

func (c *UnixConn) writeTo(b []byte, addr *UnixAddr) (int, error) {
	return 0, syscall.EPLAN9
}

func (c *UnixConn) writeMsg(b, oob []byte, addr *UnixAddr) (n, oobn int, err error) {
	return 0, 0, syscall.EPLAN9
}

func dialUnix(ctx context.Context, network string, laddr, raddr *UnixAddr) (*UnixConn, error) {
	return nil, syscall.EPLAN9
}

func (ln *UnixListener) accept() (*UnixConn, error) {
	return nil, syscall.EPLAN9
}

func (ln *UnixListener) close() error {
	return syscall.EPLAN9
}

func (ln *UnixListener) file() (*os.File, error) {
	return nil, syscall.EPLAN9
}

func listenUnix(ctx context.Context, network string, laddr *UnixAddr) (*UnixListener, error) {
	return nil, syscall.EPLAN9
}

func listenUnixgram(ctx context.Context, network string, laddr *UnixAddr) (*UnixConn, error) {
	return nil, syscall.EPLAN9
}

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"context"
	"io"
	"os"
)

func (c *TCPConn) readFrom(r io.Reader) (int64, error) {
	return genericReadFrom(c, r)
}

func (sd *sysDialer) dialTCP(ctx context.Context, laddr, raddr *TCPAddr) (*TCPConn, error) {
	if testHookDialTCP != nil {
		return testHookDialTCP(ctx, sd.network, laddr, raddr)
	}
	return sd.doDialTCP(ctx, laddr, raddr)
}

func (sd *sysDialer) doDialTCP(ctx context.Context, laddr, raddr *TCPAddr) (*TCPConn, error) {
	switch sd.network {
	case "tcp", "tcp4", "tcp6":
	default:
		return nil, UnknownNetworkError(sd.network)
	}
	if raddr == nil {
		return nil, errMissingAddress
	}
	fd, err := dialPlan9(ctx, sd.network, laddr, raddr)
	if err != nil {
		return nil, err
	}
	return newTCPConn(fd), nil
}

func (ln *TCPListener) ok() bool { return ln != nil && ln.fd != nil && ln.fd.ctl != nil }

func (ln *TCPListener) accept() (*TCPConn, error) {
	fd, err := ln.fd.acceptPlan9()
	if err != nil {
		return nil, err
	}
	return newTCPConn(fd), nil
}

func (ln *TCPListener) close() error {
	if err := ln.fd.pfd.Close(); err != nil {
		return err
	}
	if _, err := ln.fd.ctl.WriteString("hangup"); err != nil {
		ln.fd.ctl.Close()
		return err
	}
	if err := ln.fd.ctl.Close(); err != nil {
		return err
	}
	return nil
}

func (ln *TCPListener) file() (*os.File, error) {
	f, err := ln.dup()
	if err != nil {
		return nil, err
	}
	return f, nil
}

func (sl *sysListener) listenTCP(ctx context.Context, laddr *TCPAddr) (*TCPListener, error) {
	fd, err := listenPlan9(ctx, sl.network, laddr)
	if err != nil {
		return nil, err
	}
	return &TCPListener{fd}, nil
}

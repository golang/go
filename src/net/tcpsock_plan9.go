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
	case "tcp4":
		// Plan 9 doesn't complain about [::]:0->127.0.0.1, so it's up to us.
		if laddr != nil && len(laddr.IP) != 0 && laddr.IP.To4() == nil {
			return nil, &AddrError{Err: "non-IPv4 local address", Addr: laddr.String()}
		}
	case "tcp", "tcp6":
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

func (l *TCPListener) ok() bool { return l != nil && l.fd != nil && l.fd.ctl != nil }

func (l *TCPListener) accept() (*TCPConn, error) {
	fd, err := l.fd.acceptPlan9()
	if err != nil {
		return nil, err
	}
	tc := newTCPConn(fd)
	if l.lc.KeepAlive >= 0 {
		setKeepAlive(fd, true)
		ka := l.lc.KeepAlive
		if l.lc.KeepAlive == 0 {
			ka = defaultTCPKeepAlive
		}
		setKeepAlivePeriod(fd, ka)
	}
	return tc, nil
}

func (l *TCPListener) close() error {
	if err := l.fd.pfd.Close(); err != nil {
		return err
	}
	if _, err := l.fd.ctl.WriteString("hangup"); err != nil {
		l.fd.ctl.Close()
		return err
	}
	if err := l.fd.ctl.Close(); err != nil {
		return err
	}
	return nil
}

func (l *TCPListener) file() (*os.File, error) {
	f, err := l.dup()
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
	return &TCPListener{fd: fd, lc: sl.ListenConfig}, nil
}

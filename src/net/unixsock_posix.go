// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd js,wasm linux netbsd openbsd solaris windows

package net

import (
	"context"
	"errors"
	"os"
	"syscall"
)

func unixSocket(ctx context.Context, net string, laddr, raddr sockaddr, mode string, ctrlFn func(string, string, syscall.RawConn) error) (*netFD, error) {
	var sotype int
	switch net {
	case "unix":
		sotype = syscall.SOCK_STREAM
	case "unixgram":
		sotype = syscall.SOCK_DGRAM
	case "unixpacket":
		sotype = syscall.SOCK_SEQPACKET
	default:
		return nil, UnknownNetworkError(net)
	}

	switch mode {
	case "dial":
		if laddr != nil && laddr.isWildcard() {
			laddr = nil
		}
		if raddr != nil && raddr.isWildcard() {
			raddr = nil
		}
		if raddr == nil && (sotype != syscall.SOCK_DGRAM || laddr == nil) {
			return nil, errMissingAddress
		}
	case "listen":
	default:
		return nil, errors.New("unknown mode: " + mode)
	}

	fd, err := socket(ctx, net, syscall.AF_UNIX, sotype, 0, false, laddr, raddr, ctrlFn)
	if err != nil {
		return nil, err
	}
	return fd, nil
}

func sockaddrToUnix(sa syscall.Sockaddr) Addr {
	if s, ok := sa.(*syscall.SockaddrUnix); ok {
		return &UnixAddr{Name: s.Name, Net: "unix"}
	}
	return nil
}

func sockaddrToUnixgram(sa syscall.Sockaddr) Addr {
	if s, ok := sa.(*syscall.SockaddrUnix); ok {
		return &UnixAddr{Name: s.Name, Net: "unixgram"}
	}
	return nil
}

func sockaddrToUnixpacket(sa syscall.Sockaddr) Addr {
	if s, ok := sa.(*syscall.SockaddrUnix); ok {
		return &UnixAddr{Name: s.Name, Net: "unixpacket"}
	}
	return nil
}

func sotypeToNet(sotype int) string {
	switch sotype {
	case syscall.SOCK_STREAM:
		return "unix"
	case syscall.SOCK_DGRAM:
		return "unixgram"
	case syscall.SOCK_SEQPACKET:
		return "unixpacket"
	default:
		panic("sotypeToNet unknown socket type")
	}
}

func (a *UnixAddr) family() int {
	return syscall.AF_UNIX
}

func (a *UnixAddr) sockaddr(family int) (syscall.Sockaddr, error) {
	if a == nil {
		return nil, nil
	}
	return &syscall.SockaddrUnix{Name: a.Name}, nil
}

func (a *UnixAddr) toLocal(net string) sockaddr {
	return a
}

func (c *UnixConn) readFrom(b []byte) (int, *UnixAddr, error) {
	var addr *UnixAddr
	n, sa, err := c.fd.readFrom(b)
	switch sa := sa.(type) {
	case *syscall.SockaddrUnix:
		if sa.Name != "" {
			addr = &UnixAddr{Name: sa.Name, Net: sotypeToNet(c.fd.sotype)}
		}
	}
	return n, addr, err
}

func (c *UnixConn) readMsg(b, oob []byte) (n, oobn, flags int, addr *UnixAddr, err error) {
	var sa syscall.Sockaddr
	n, oobn, flags, sa, err = c.fd.readMsg(b, oob)
	switch sa := sa.(type) {
	case *syscall.SockaddrUnix:
		if sa.Name != "" {
			addr = &UnixAddr{Name: sa.Name, Net: sotypeToNet(c.fd.sotype)}
		}
	}
	return
}

func (c *UnixConn) writeTo(b []byte, addr *UnixAddr) (int, error) {
	if c.fd.isConnected {
		return 0, ErrWriteToConnected
	}
	if addr == nil {
		return 0, errMissingAddress
	}
	if addr.Net != sotypeToNet(c.fd.sotype) {
		return 0, syscall.EAFNOSUPPORT
	}
	sa := &syscall.SockaddrUnix{Name: addr.Name}
	return c.fd.writeTo(b, sa)
}

func (c *UnixConn) writeMsg(b, oob []byte, addr *UnixAddr) (n, oobn int, err error) {
	if c.fd.sotype == syscall.SOCK_DGRAM && c.fd.isConnected {
		return 0, 0, ErrWriteToConnected
	}
	var sa syscall.Sockaddr
	if addr != nil {
		if addr.Net != sotypeToNet(c.fd.sotype) {
			return 0, 0, syscall.EAFNOSUPPORT
		}
		sa = &syscall.SockaddrUnix{Name: addr.Name}
	}
	return c.fd.writeMsg(b, oob, sa)
}

func (sd *sysDialer) dialUnix(ctx context.Context, laddr, raddr *UnixAddr) (*UnixConn, error) {
	fd, err := unixSocket(ctx, sd.network, laddr, raddr, "dial", sd.Dialer.Control)
	if err != nil {
		return nil, err
	}
	return newUnixConn(fd), nil
}

func (ln *UnixListener) accept() (*UnixConn, error) {
	fd, err := ln.fd.accept()
	if err != nil {
		return nil, err
	}
	return newUnixConn(fd), nil
}

func (ln *UnixListener) close() error {
	// The operating system doesn't clean up
	// the file that announcing created, so
	// we have to clean it up ourselves.
	// There's a race here--we can't know for
	// sure whether someone else has come along
	// and replaced our socket name already--
	// but this sequence (remove then close)
	// is at least compatible with the auto-remove
	// sequence in ListenUnix. It's only non-Go
	// programs that can mess us up.
	// Even if there are racy calls to Close, we want to unlink only for the first one.
	ln.unlinkOnce.Do(func() {
		if ln.path[0] != '@' && ln.unlink {
			syscall.Unlink(ln.path)
		}
	})
	return ln.fd.Close()
}

func (ln *UnixListener) file() (*os.File, error) {
	f, err := ln.fd.dup()
	if err != nil {
		return nil, err
	}
	return f, nil
}

// SetUnlinkOnClose sets whether the underlying socket file should be removed
// from the file system when the listener is closed.
//
// The default behavior is to unlink the socket file only when package net created it.
// That is, when the listener and the underlying socket file were created by a call to
// Listen or ListenUnix, then by default closing the listener will remove the socket file.
// but if the listener was created by a call to FileListener to use an already existing
// socket file, then by default closing the listener will not remove the socket file.
func (l *UnixListener) SetUnlinkOnClose(unlink bool) {
	l.unlink = unlink
}

func (sl *sysListener) listenUnix(ctx context.Context, laddr *UnixAddr) (*UnixListener, error) {
	fd, err := unixSocket(ctx, sl.network, laddr, nil, "listen", sl.ListenConfig.Control)
	if err != nil {
		return nil, err
	}
	return &UnixListener{fd: fd, path: fd.laddr.String(), unlink: true}, nil
}

func (sl *sysListener) listenUnixgram(ctx context.Context, laddr *UnixAddr) (*UnixConn, error) {
	fd, err := unixSocket(ctx, sl.network, laddr, nil, "listen", sl.ListenConfig.Control)
	if err != nil {
		return nil, err
	}
	return newUnixConn(fd), nil
}

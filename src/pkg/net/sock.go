// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Sockets

package net

import (
	"os";
	"reflect";
	"syscall";
)

// Boolean to int.
func boolint(b bool) int {
	if b {
		return 1
	}
	return 0
}

// Generic socket creation.
func socket(net, laddr, raddr string, f, p, t int, la, ra syscall.Sockaddr) (fd *netFD, err os.Error) {
	// See ../syscall/exec.go for description of ForkLock.
	syscall.ForkLock.RLock();
	s, e := syscall.Socket(f, p, t);
	if e != 0 {
		syscall.ForkLock.RUnlock();
		return nil, os.Errno(e)
	}
	syscall.CloseOnExec(s);
	syscall.ForkLock.RUnlock();

	// Allow reuse of recently-used addresses.
	syscall.SetsockoptInt(s, syscall.SOL_SOCKET, syscall.SO_REUSEADDR, 1);

	if la != nil {
		e = syscall.Bind(s, la);
		if e != 0 {
			syscall.Close(s);
			return nil, os.Errno(e)
		}
	}

	if ra != nil {
		e = syscall.Connect(s, ra);
		if e != 0 {
			syscall.Close(s);
			return nil, os.Errno(e)
		}
	}

	fd, err = newFD(s, net, laddr, raddr);
	if err != nil {
		syscall.Close(s);
		return nil, err
	}

	return fd, nil
}


// Generic implementation of Conn interface; not exported.
type connBase struct {
	fd *netFD;
	raddr string;
}

func (c *connBase) LocalAddr() string {
	if c == nil {
		return ""
	}
	return c.fd.addr();
}

func (c *connBase) RemoteAddr() string {
	if c == nil {
		return ""
	}
	return c.fd.remoteAddr();
}

func (c *connBase) File() *os.File {
	if c == nil {
		return nil
	}
	return c.fd.file;
}

func (c *connBase) sysFD() int {
	if c == nil || c.fd == nil {
		return -1;
	}
	return c.fd.fd;
}

func (c *connBase) Read(b []byte) (n int, err os.Error) {
	n, err = c.fd.Read(b);
	return n, err
}

func (c *connBase) Write(b []byte) (n int, err os.Error) {
	n, err = c.fd.Write(b);
	return n, err
}

func (c *connBase) ReadFrom(b []byte) (n int, raddr string, err os.Error) {
	if c == nil {
		return -1, "", os.EINVAL
	}
	n, err = c.Read(b);
	return n, c.raddr, err
}

func (c *connBase) WriteTo(raddr string, b []byte) (n int, err os.Error) {
	if c == nil {
		return -1, os.EINVAL
	}
	if raddr != c.raddr {
		return -1, os.EINVAL
	}
	n, err = c.Write(b);
	return n, err
}

func (c *connBase) Close() os.Error {
	if c == nil {
		return os.EINVAL
	}
	return c.fd.Close()
}


func setsockoptInt(fd, level, opt int, value int) os.Error {
	return os.NewSyscallError("setsockopt", syscall.SetsockoptInt(fd, level, opt, value));
}

func setsockoptNsec(fd, level, opt int, nsec int64) os.Error {
	var tv = syscall.NsecToTimeval(nsec);
	return os.NewSyscallError("setsockopt", syscall.SetsockoptTimeval(fd, level, opt, &tv));
}

func (c *connBase) SetReadBuffer(bytes int) os.Error {
	return setsockoptInt(c.sysFD(), syscall.SOL_SOCKET, syscall.SO_RCVBUF, bytes);
}

func (c *connBase) SetWriteBuffer(bytes int) os.Error {
	return setsockoptInt(c.sysFD(), syscall.SOL_SOCKET, syscall.SO_SNDBUF, bytes);
}

func (c *connBase) SetReadTimeout(nsec int64) os.Error {
	c.fd.rdeadline_delta = nsec;
	return nil;
}

func (c *connBase) SetWriteTimeout(nsec int64) os.Error {
	c.fd.wdeadline_delta = nsec;
	return nil;
}

func (c *connBase) SetTimeout(nsec int64) os.Error {
	if e := c.SetReadTimeout(nsec); e != nil {
		return e
	}
	return c.SetWriteTimeout(nsec)
}

func (c *connBase) SetReuseAddr(reuse bool) os.Error {
	return setsockoptInt(c.sysFD(), syscall.SOL_SOCKET, syscall.SO_REUSEADDR, boolint(reuse));
}

func (c *connBase) BindToDevice(dev string) os.Error {
	// TODO(rsc): call setsockopt with null-terminated string pointer
	return os.EINVAL
}

func (c *connBase) SetDontRoute(dontroute bool) os.Error {
	return setsockoptInt(c.sysFD(), syscall.SOL_SOCKET, syscall.SO_DONTROUTE, boolint(dontroute));
}

func (c *connBase) SetKeepAlive(keepalive bool) os.Error {
	return setsockoptInt(c.sysFD(), syscall.SOL_SOCKET, syscall.SO_KEEPALIVE, boolint(keepalive));
}

func (c *connBase) SetLinger(sec int) os.Error {
	var l syscall.Linger;
	if sec >= 0 {
		l.Onoff = 1;
		l.Linger = int32(sec);
	} else {
		l.Onoff = 0;
		l.Linger = 0;
	}
	e := syscall.SetsockoptLinger(c.sysFD(), syscall.SOL_SOCKET, syscall.SO_LINGER, &l);
	return os.NewSyscallError("setsockopt", e);
}


type UnknownSocketError struct {
	sa syscall.Sockaddr;
}
func (e *UnknownSocketError) String() string {
	return "unknown socket address type " + reflect.Typeof(e.sa).String()
}

func sockaddrToString(sa syscall.Sockaddr) (name string, err os.Error) {
	switch a := sa.(type) {
	case *syscall.SockaddrInet4:
		return joinHostPort(IP(&a.Addr).String(), itoa(a.Port)), nil;
	case *syscall.SockaddrInet6:
		return joinHostPort(IP(&a.Addr).String(), itoa(a.Port)), nil;
	case *syscall.SockaddrUnix:
		return a.Name, nil;
	}

	return "", &UnknownSocketError{sa};
}


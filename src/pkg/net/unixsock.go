// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Unix domain sockets

package net

import (
	"os";
	"syscall";
)

func unixSocket(net, laddr, raddr string, mode string) (fd *netFD, err os.Error) {
	var proto int;
	switch net {
	default:
		return nil, UnknownNetworkError(net);
	case "unix":
		proto = syscall.SOCK_STREAM;
	case "unix-dgram":
		proto = syscall.SOCK_DGRAM;
	}

	var la, ra syscall.Sockaddr;
	switch mode {
	default:
		panic("unixSocket", mode);

	case "dial":
		if laddr != "" {
			return nil, &OpError{mode, net, raddr, &AddrError{"unexpected local address", laddr}}
		}
		if raddr == "" {
			return nil, &OpError{mode, net, "", errMissingAddress}
		}
		ra = &syscall.SockaddrUnix{Name: raddr};

	case "listen":
		if laddr == "" {
			return nil, &OpError{mode, net, "", errMissingAddress}
		}
		la = &syscall.SockaddrUnix{Name: laddr};
		if raddr != "" {
			return nil, &OpError{mode, net, laddr, &AddrError{"unexpected remote address", raddr}}
		}
	}

	fd, err = socket(net, laddr, raddr, syscall.AF_UNIX, proto, 0, la, ra);
	if err != nil {
		goto Error;
	}
	return fd, nil;

Error:
	addr := raddr;
	if mode == "listen" {
		addr = laddr;
	}
	return nil, &OpError{mode, net, addr, err};
}

// ConnUnix is an implementation of the Conn interface
// for connections to Unix domain sockets.
type ConnUnix struct {
	connBase
}

func newConnUnix(fd *netFD, raddr string) *ConnUnix {
	c := new(ConnUnix);
	c.fd = fd;
	c.raddr = raddr;
	return c;
}

// DialUnix is like Dial but can only connect to Unix domain sockets
// and returns a ConnUnix structure.  The laddr argument must be
// the empty string; it is included only to match the signature of
// the other dial routines.
func DialUnix(net, laddr, raddr string) (c *ConnUnix, err os.Error) {
	fd, e := unixSocket(net, laddr, raddr, "dial");
	if e != nil {
		return nil, e
	}
	return newConnUnix(fd, raddr), nil;
}

// ListenerUnix is a Unix domain socket listener.
// Clients should typically use variables of type Listener
// instead of assuming Unix domain sockets.
type ListenerUnix struct {
	fd *netFD;
	laddr string
}

// ListenUnix announces on the Unix domain socket laddr and returns a Unix listener.
// Net can be either "unix" (stream sockets) or "unix-dgram" (datagram sockets).
func ListenUnix(net, laddr string) (l *ListenerUnix, err os.Error) {
	fd, e := unixSocket(net, laddr, "", "listen");
	if e != nil {
		if pe, ok := e.(*os.PathError); ok {
			e = pe.Error;
		}
		// Check for socket ``in use'' but ``refusing connections,''
		// which means some program created it and exited
		// without unlinking it from the file system.
		// Clean up on that program's behalf and try again.
		// Don't do this for Linux's ``abstract'' sockets, which begin with @.
		if e != os.EADDRINUSE || laddr[0] == '@' {
			return nil, e;
		}
		fd1, e1 := unixSocket(net, "", laddr, "dial");
		if e1 == nil {
			fd1.Close();
		}
		if pe, ok := e1.(*os.PathError); ok {
			e1 = pe.Error;
		}
		if e1 != os.ECONNREFUSED {
			return nil, e;
		}
		syscall.Unlink(laddr);
		fd1, e1 = unixSocket(net, laddr, "", "listen");
		if e1 != nil {
			return nil, e;
		}
		fd = fd1;
	}
	e1 := syscall.Listen(fd.fd, 8); // listenBacklog());
	if e1 != 0 {
		syscall.Close(fd.fd);
		return nil, &OpError{"listen", "unix", laddr, os.Errno(e1)};
	}
	return &ListenerUnix{fd, laddr}, nil;
}

// AcceptUnix accepts the next incoming call and returns the new connection
// and the remote address.
func (l *ListenerUnix) AcceptUnix() (c *ConnUnix, raddr string, err os.Error) {
	if l == nil || l.fd == nil || l.fd.fd < 0 {
		return nil, "", os.EINVAL
	}
	fd, e := l.fd.accept();
	if e != nil {
		return nil, "", e
	}
	return newConnUnix(fd, fd.raddr), raddr, nil
}

// Accept implements the Accept method in the Listener interface;
// it waits for the next call and returns a generic Conn.
func (l *ListenerUnix) Accept() (c Conn, raddr string, err os.Error) {
	// TODO(rsc): Should return l.AcceptUnix() be okay here?
	// There is a type conversion -- the first return arg of
	// l.AcceptUnix() is *ConnUnix and it gets converted to Conn
	// in the explicit assignment.
	c, raddr, err = l.AcceptUnix();
	return;
}


// Close stops listening on the Unix address.
// Already accepted connections are not closed.
func (l *ListenerUnix) Close() os.Error {
	if l == nil || l.fd == nil {
		return os.EINVAL
	}

	// The operating system doesn't clean up
	// the file that announcing created, so
	// we have to clean it up ourselves.
	// There's a race here--we can't know for
	// sure whether someone else has come along
	// and replaced our socket name already--
	// but this sequence (remove then close)
	// is at least compatible with the auto-remove
	// sequence in ListenUnix.  It's only non-Go
	// programs that can mess us up.
	if l.laddr[0] != '@' {
		syscall.Unlink(l.laddr);
	}
	err := l.fd.Close();
	l.fd = nil;
	return err;
}

// Addr returns the listener's network address.
func (l *ListenerUnix) Addr() string {
	return l.fd.addr();
}

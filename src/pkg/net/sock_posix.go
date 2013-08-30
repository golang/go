// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd windows

package net

import (
	"os"
	"syscall"
	"time"
)

// A sockaddr represents a TCP, UDP, IP or Unix network endpoint
// address that can be converted into a syscall.Sockaddr.
type sockaddr interface {
	Addr

	netaddr

	// family returns the platform-dependent address family
	// identifier.
	family() int

	// isWildcard reports whether the address is a wildcard
	// address.
	isWildcard() bool

	// sockaddr returns the address converted into a syscall
	// sockaddr type that implements syscall.Sockaddr
	// interface. It returns a nil interface when the address is
	// nil.
	sockaddr(family int) (syscall.Sockaddr, error)
}

// socket returns a network file descriptor that is ready for
// asynchronous I/O using the network poller.
func socket(net string, family, sotype, proto int, ipv6only bool, laddr, raddr sockaddr, deadline time.Time, toAddr func(syscall.Sockaddr) Addr) (fd *netFD, err error) {
	s, err := sysSocket(family, sotype, proto)
	if err != nil {
		return nil, err
	}
	if err = setDefaultSockopts(s, family, sotype, ipv6only); err != nil {
		closesocket(s)
		return nil, err
	}
	if fd, err = newFD(s, family, sotype, net); err != nil {
		closesocket(s)
		return nil, err
	}

	// This function makes a network file descriptor for the
	// following applications:
	//
	// - An endpoint holder that opens a passive stream
	//   connenction, known as a stream listener
	//
	// - An endpoint holder that opens a destination-unspecific
	//   datagram connection, known as a datagram listener
	//
	// - An endpoint holder that opens an active stream or a
	//   destination-specific datagram connection, known as a
	//   dialer
	//
	// - An endpoint holder that opens the other connection, such
	//   as talking to the protocol stack inside the kernel
	//
	// For stream and datagram listeners, they will only require
	// named sockets, so we can assume that it's just a request
	// from stream or datagram listeners when laddr is not nil but
	// raddr is nil. Otherwise we assume it's just for dialers or
	// the other connection holders.

	if laddr != nil && raddr == nil {
		switch sotype {
		case syscall.SOCK_STREAM, syscall.SOCK_SEQPACKET:
			if err := fd.listenStream(laddr, listenerBacklog, toAddr); err != nil {
				fd.Close()
				return nil, err
			}
			return fd, nil
		case syscall.SOCK_DGRAM:
			if err := fd.listenDatagram(laddr, toAddr); err != nil {
				fd.Close()
				return nil, err
			}
			return fd, nil
		}
	}
	if err := fd.dial(laddr, raddr, deadline, toAddr); err != nil {
		fd.Close()
		return nil, err
	}
	return fd, nil
}

func (fd *netFD) dial(laddr, raddr sockaddr, deadline time.Time, toAddr func(syscall.Sockaddr) Addr) error {
	var err error
	var lsa syscall.Sockaddr
	if laddr != nil {
		if lsa, err = laddr.sockaddr(fd.family); err != nil {
			return err
		} else if lsa != nil {
			if err := syscall.Bind(fd.sysfd, lsa); err != nil {
				return os.NewSyscallError("bind", err)
			}
		}
	}
	if err := fd.init(); err != nil {
		return err
	}
	var rsa syscall.Sockaddr
	if raddr != nil {
		if rsa, err = raddr.sockaddr(fd.family); err != nil {
			return err
		} else if rsa != nil {
			if !deadline.IsZero() {
				fd.setWriteDeadline(deadline)
			}
			if err := fd.connect(lsa, rsa); err != nil {
				return err
			}
			fd.isConnected = true
			if !deadline.IsZero() {
				fd.setWriteDeadline(noDeadline)
			}
		}
	}
	lsa, _ = syscall.Getsockname(fd.sysfd)
	if rsa, _ = syscall.Getpeername(fd.sysfd); rsa != nil {
		fd.setAddr(toAddr(lsa), toAddr(rsa))
	} else {
		fd.setAddr(toAddr(lsa), raddr)
	}
	return nil
}

func (fd *netFD) listenStream(laddr sockaddr, backlog int, toAddr func(syscall.Sockaddr) Addr) error {
	if err := setDefaultListenerSockopts(fd.sysfd); err != nil {
		return err
	}
	if lsa, err := laddr.sockaddr(fd.family); err != nil {
		return err
	} else if lsa != nil {
		if err := syscall.Bind(fd.sysfd, lsa); err != nil {
			return os.NewSyscallError("bind", err)
		}
	}
	if err := syscall.Listen(fd.sysfd, backlog); err != nil {
		return os.NewSyscallError("listen", err)
	}
	if err := fd.init(); err != nil {
		return err
	}
	lsa, _ := syscall.Getsockname(fd.sysfd)
	fd.setAddr(toAddr(lsa), nil)
	return nil
}

func (fd *netFD) listenDatagram(laddr sockaddr, toAddr func(syscall.Sockaddr) Addr) error {
	switch addr := laddr.(type) {
	case *UDPAddr:
		// We provide a socket that listens to a wildcard
		// address with reusable UDP port when the given laddr
		// is an appropriate UDP multicast address prefix.
		// This makes it possible for a single UDP listener to
		// join multiple different group addresses, for
		// multiple UDP listeners that listen on the same UDP
		// port to join the same group address.
		if addr.IP != nil && addr.IP.IsMulticast() {
			if err := setDefaultMulticastSockopts(fd.sysfd); err != nil {
				return err
			}
			addr := *addr
			switch fd.family {
			case syscall.AF_INET:
				addr.IP = IPv4zero
			case syscall.AF_INET6:
				addr.IP = IPv6unspecified
			}
			laddr = &addr
		}
	}
	if lsa, err := laddr.sockaddr(fd.family); err != nil {
		return err
	} else if lsa != nil {
		if err := syscall.Bind(fd.sysfd, lsa); err != nil {
			return os.NewSyscallError("bind", err)
		}
	}
	if err := fd.init(); err != nil {
		return err
	}
	lsa, _ := syscall.Getsockname(fd.sysfd)
	fd.setAddr(toAddr(lsa), nil)
	return nil
}

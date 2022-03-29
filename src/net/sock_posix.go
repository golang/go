// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris || windows

package net

import (
	"context"
	"internal/poll"
	"os"
	"syscall"
)

// socket returns a network file descriptor that is ready for
// asynchronous I/O using the network poller.
func socket(ctx context.Context, net string, family, sotype, proto int, ipv6only bool, laddr, raddr sockaddr, ctrlFn func(string, string, syscall.RawConn) error) (fd *netFD, err error) {
	s, err := sysSocket(family, sotype, proto)
	if err != nil {
		return nil, err
	}
	if err = setDefaultSockopts(s, family, sotype, ipv6only); err != nil {
		poll.CloseFunc(s)
		return nil, err
	}
	if fd, err = newFD(s, family, sotype, net); err != nil {
		poll.CloseFunc(s)
		return nil, err
	}

	// This function makes a network file descriptor for the
	// following applications:
	//
	// - An endpoint holder that opens a passive stream
	//   connection, known as a stream listener
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
			if err := fd.listenStream(laddr, listenerBacklog(), ctrlFn); err != nil {
				fd.Close()
				return nil, err
			}
			return fd, nil
		case syscall.SOCK_DGRAM:
			if err := fd.listenDatagram(laddr, ctrlFn); err != nil {
				fd.Close()
				return nil, err
			}
			return fd, nil
		}
	}
	if err := fd.dial(ctx, laddr, raddr, ctrlFn); err != nil {
		fd.Close()
		return nil, err
	}
	return fd, nil
}

func (fd *netFD) ctrlNetwork() string {
	switch fd.net {
	case "unix", "unixgram", "unixpacket":
		return fd.net
	}
	switch fd.net[len(fd.net)-1] {
	case '4', '6':
		return fd.net
	}
	if fd.family == syscall.AF_INET {
		return fd.net + "4"
	}
	return fd.net + "6"
}

func (fd *netFD) addrFunc() func(syscall.Sockaddr) Addr {
	switch fd.family {
	case syscall.AF_INET, syscall.AF_INET6:
		switch fd.sotype {
		case syscall.SOCK_STREAM:
			return sockaddrToTCP
		case syscall.SOCK_DGRAM:
			return sockaddrToUDP
		case syscall.SOCK_RAW:
			return sockaddrToIP
		}
	case syscall.AF_UNIX:
		switch fd.sotype {
		case syscall.SOCK_STREAM:
			return sockaddrToUnix
		case syscall.SOCK_DGRAM:
			return sockaddrToUnixgram
		case syscall.SOCK_SEQPACKET:
			return sockaddrToUnixpacket
		}
	}
	return func(syscall.Sockaddr) Addr { return nil }
}

func (fd *netFD) dial(ctx context.Context, laddr, raddr sockaddr, ctrlFn func(string, string, syscall.RawConn) error) error {
	if ctrlFn != nil {
		c, err := newRawConn(fd)
		if err != nil {
			return err
		}
		var ctrlAddr string
		if raddr != nil {
			ctrlAddr = raddr.String()
		} else if laddr != nil {
			ctrlAddr = laddr.String()
		}
		if err := ctrlFn(fd.ctrlNetwork(), ctrlAddr, c); err != nil {
			return err
		}
	}
	var err error
	var lsa syscall.Sockaddr
	if laddr != nil {
		if lsa, err = laddr.sockaddr(fd.family); err != nil {
			return err
		} else if lsa != nil {
			if err = syscall.Bind(fd.pfd.Sysfd, lsa); err != nil {
				return os.NewSyscallError("bind", err)
			}
		}
	}
	var rsa syscall.Sockaddr  // remote address from the user
	var crsa syscall.Sockaddr // remote address we actually connected to
	if raddr != nil {
		if rsa, err = raddr.sockaddr(fd.family); err != nil {
			return err
		}
		if crsa, err = fd.connect(ctx, lsa, rsa); err != nil {
			return err
		}
		fd.isConnected = true
	} else {
		if err := fd.init(); err != nil {
			return err
		}
	}
	// Record the local and remote addresses from the actual socket.
	// Get the local address by calling Getsockname.
	// For the remote address, use
	// 1) the one returned by the connect method, if any; or
	// 2) the one from Getpeername, if it succeeds; or
	// 3) the one passed to us as the raddr parameter.
	lsa, _ = syscall.Getsockname(fd.pfd.Sysfd)
	if crsa != nil {
		fd.setAddr(fd.addrFunc()(lsa), fd.addrFunc()(crsa))
	} else if rsa, _ = syscall.Getpeername(fd.pfd.Sysfd); rsa != nil {
		fd.setAddr(fd.addrFunc()(lsa), fd.addrFunc()(rsa))
	} else {
		fd.setAddr(fd.addrFunc()(lsa), raddr)
	}
	return nil
}

func (fd *netFD) listenStream(laddr sockaddr, backlog int, ctrlFn func(string, string, syscall.RawConn) error) error {
	var err error
	if err = setDefaultListenerSockopts(fd.pfd.Sysfd); err != nil {
		return err
	}
	var lsa syscall.Sockaddr
	if lsa, err = laddr.sockaddr(fd.family); err != nil {
		return err
	}
	if ctrlFn != nil {
		c, err := newRawConn(fd)
		if err != nil {
			return err
		}
		if err := ctrlFn(fd.ctrlNetwork(), laddr.String(), c); err != nil {
			return err
		}
	}
	if err = syscall.Bind(fd.pfd.Sysfd, lsa); err != nil {
		return os.NewSyscallError("bind", err)
	}
	if err = listenFunc(fd.pfd.Sysfd, backlog); err != nil {
		return os.NewSyscallError("listen", err)
	}
	if err = fd.init(); err != nil {
		return err
	}
	lsa, _ = syscall.Getsockname(fd.pfd.Sysfd)
	fd.setAddr(fd.addrFunc()(lsa), nil)
	return nil
}

func (fd *netFD) listenDatagram(laddr sockaddr, ctrlFn func(string, string, syscall.RawConn) error) error {
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
			if err := setDefaultMulticastSockopts(fd.pfd.Sysfd); err != nil {
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
	var err error
	var lsa syscall.Sockaddr
	if lsa, err = laddr.sockaddr(fd.family); err != nil {
		return err
	}
	if ctrlFn != nil {
		c, err := newRawConn(fd)
		if err != nil {
			return err
		}
		if err := ctrlFn(fd.ctrlNetwork(), laddr.String(), c); err != nil {
			return err
		}
	}
	if err = syscall.Bind(fd.pfd.Sysfd, lsa); err != nil {
		return os.NewSyscallError("bind", err)
	}
	if err = fd.init(); err != nil {
		return err
	}
	lsa, _ = syscall.Getsockname(fd.pfd.Sysfd)
	fd.setAddr(fd.addrFunc()(lsa), nil)
	return nil
}

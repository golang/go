// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd windows

// Socket options

package net

import (
	"bytes"
	"os"
	"syscall"
	"time"
)

// Boolean to int.
func boolint(b bool) int {
	if b {
		return 1
	}
	return 0
}

func ipv4AddrToInterface(ip IP) (*Interface, error) {
	ift, err := Interfaces()
	if err != nil {
		return nil, err
	}
	for _, ifi := range ift {
		ifat, err := ifi.Addrs()
		if err != nil {
			return nil, err
		}
		for _, ifa := range ifat {
			switch v := ifa.(type) {
			case *IPAddr:
				if ip.Equal(v.IP) {
					return &ifi, nil
				}
			case *IPNet:
				if ip.Equal(v.IP) {
					return &ifi, nil
				}
			}
		}
	}
	if ip.Equal(IPv4zero) {
		return nil, nil
	}
	return nil, errNoSuchInterface
}

func interfaceToIPv4Addr(ifi *Interface) (IP, error) {
	if ifi == nil {
		return IPv4zero, nil
	}
	ifat, err := ifi.Addrs()
	if err != nil {
		return nil, err
	}
	for _, ifa := range ifat {
		switch v := ifa.(type) {
		case *IPAddr:
			if v.IP.To4() != nil {
				return v.IP, nil
			}
		case *IPNet:
			if v.IP.To4() != nil {
				return v.IP, nil
			}
		}
	}
	return nil, errNoSuchInterface
}

func setIPv4MreqToInterface(mreq *syscall.IPMreq, ifi *Interface) error {
	if ifi == nil {
		return nil
	}
	ifat, err := ifi.Addrs()
	if err != nil {
		return err
	}
	for _, ifa := range ifat {
		switch v := ifa.(type) {
		case *IPAddr:
			if a := v.IP.To4(); a != nil {
				copy(mreq.Interface[:], a)
				goto done
			}
		case *IPNet:
			if a := v.IP.To4(); a != nil {
				copy(mreq.Interface[:], a)
				goto done
			}
		}
	}
done:
	if bytes.Equal(mreq.Multiaddr[:], IPv4zero.To4()) {
		return errNoSuchMulticastInterface
	}
	return nil
}

func setReadBuffer(fd *netFD, bytes int) error {
	fd.incref()
	defer fd.decref()
	return os.NewSyscallError("setsockopt", syscall.SetsockoptInt(fd.sysfd, syscall.SOL_SOCKET, syscall.SO_RCVBUF, bytes))
}

func setWriteBuffer(fd *netFD, bytes int) error {
	fd.incref()
	defer fd.decref()
	return os.NewSyscallError("setsockopt", syscall.SetsockoptInt(fd.sysfd, syscall.SOL_SOCKET, syscall.SO_SNDBUF, bytes))
}

func setReadDeadline(fd *netFD, t time.Time) error {
	if t.IsZero() {
		fd.rdeadline = 0
	} else {
		fd.rdeadline = t.UnixNano()
	}
	return nil
}

func setWriteDeadline(fd *netFD, t time.Time) error {
	if t.IsZero() {
		fd.wdeadline = 0
	} else {
		fd.wdeadline = t.UnixNano()
	}
	return nil
}

func setDeadline(fd *netFD, t time.Time) error {
	if err := setReadDeadline(fd, t); err != nil {
		return err
	}
	return setWriteDeadline(fd, t)
}

func setReuseAddr(fd *netFD, reuse bool) error {
	fd.incref()
	defer fd.decref()
	return os.NewSyscallError("setsockopt", syscall.SetsockoptInt(fd.sysfd, syscall.SOL_SOCKET, syscall.SO_REUSEADDR, boolint(reuse)))
}

func setDontRoute(fd *netFD, dontroute bool) error {
	fd.incref()
	defer fd.decref()
	return os.NewSyscallError("setsockopt", syscall.SetsockoptInt(fd.sysfd, syscall.SOL_SOCKET, syscall.SO_DONTROUTE, boolint(dontroute)))
}

func setKeepAlive(fd *netFD, keepalive bool) error {
	fd.incref()
	defer fd.decref()
	return os.NewSyscallError("setsockopt", syscall.SetsockoptInt(fd.sysfd, syscall.SOL_SOCKET, syscall.SO_KEEPALIVE, boolint(keepalive)))
}

func setNoDelay(fd *netFD, noDelay bool) error {
	fd.incref()
	defer fd.decref()
	return os.NewSyscallError("setsockopt", syscall.SetsockoptInt(fd.sysfd, syscall.IPPROTO_TCP, syscall.TCP_NODELAY, boolint(noDelay)))
}

func setLinger(fd *netFD, sec int) error {
	var l syscall.Linger
	if sec >= 0 {
		l.Onoff = 1
		l.Linger = int32(sec)
	} else {
		l.Onoff = 0
		l.Linger = 0
	}
	fd.incref()
	defer fd.decref()
	return os.NewSyscallError("setsockopt", syscall.SetsockoptLinger(fd.sysfd, syscall.SOL_SOCKET, syscall.SO_LINGER, &l))
}

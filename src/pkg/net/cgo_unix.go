// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !netgo
// +build darwin dragonfly freebsd linux netbsd openbsd

package net

/*
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
*/
import "C"

import (
	"syscall"
	"unsafe"
)

func cgoLookupHost(name string) (addrs []string, err error, completed bool) {
	ip, err, completed := cgoLookupIP(name)
	for _, p := range ip {
		addrs = append(addrs, p.String())
	}
	return
}

func cgoLookupPort(net, service string) (port int, err error, completed bool) {
	acquireThread()
	defer releaseThread()

	var res *C.struct_addrinfo
	var hints C.struct_addrinfo

	switch net {
	case "":
		// no hints
	case "tcp", "tcp4", "tcp6":
		hints.ai_socktype = C.SOCK_STREAM
		hints.ai_protocol = C.IPPROTO_TCP
	case "udp", "udp4", "udp6":
		hints.ai_socktype = C.SOCK_DGRAM
		hints.ai_protocol = C.IPPROTO_UDP
	default:
		return 0, UnknownNetworkError(net), true
	}
	if len(net) >= 4 {
		switch net[3] {
		case '4':
			hints.ai_family = C.AF_INET
		case '6':
			hints.ai_family = C.AF_INET6
		}
	}

	s := C.CString(service)
	defer C.free(unsafe.Pointer(s))
	if C.getaddrinfo(nil, s, &hints, &res) == 0 {
		defer C.freeaddrinfo(res)
		for r := res; r != nil; r = r.ai_next {
			switch r.ai_family {
			default:
				continue
			case C.AF_INET:
				sa := (*syscall.RawSockaddrInet4)(unsafe.Pointer(r.ai_addr))
				p := (*[2]byte)(unsafe.Pointer(&sa.Port))
				return int(p[0])<<8 | int(p[1]), nil, true
			case C.AF_INET6:
				sa := (*syscall.RawSockaddrInet6)(unsafe.Pointer(r.ai_addr))
				p := (*[2]byte)(unsafe.Pointer(&sa.Port))
				return int(p[0])<<8 | int(p[1]), nil, true
			}
		}
	}
	return 0, &AddrError{"unknown port", net + "/" + service}, true
}

func cgoLookupIPCNAME(name string) (addrs []IP, cname string, err error, completed bool) {
	acquireThread()
	defer releaseThread()

	var res *C.struct_addrinfo
	var hints C.struct_addrinfo

	hints.ai_flags = cgoAddrInfoFlags()
	hints.ai_socktype = C.SOCK_STREAM

	h := C.CString(name)
	defer C.free(unsafe.Pointer(h))
	gerrno, err := C.getaddrinfo(h, nil, &hints, &res)
	if gerrno != 0 {
		var str string
		if gerrno == C.EAI_NONAME {
			str = noSuchHost
		} else if gerrno == C.EAI_SYSTEM {
			if err == nil {
				// err should not be nil, but sometimes getaddrinfo returns
				// gerrno == C.EAI_SYSTEM with err == nil on Linux.
				// The report claims that it happens when we have too many
				// open files, so use syscall.EMFILE (too many open files in system).
				// Most system calls would return ENFILE (too many open files),
				// so at the least EMFILE should be easy to recognize if this
				// comes up again. golang.org/issue/6232.
				err = syscall.EMFILE
			}
			str = err.Error()
		} else {
			str = C.GoString(C.gai_strerror(gerrno))
		}
		return nil, "", &DNSError{Err: str, Name: name}, true
	}
	defer C.freeaddrinfo(res)
	if res != nil {
		cname = C.GoString(res.ai_canonname)
		if cname == "" {
			cname = name
		}
		if len(cname) > 0 && cname[len(cname)-1] != '.' {
			cname += "."
		}
	}
	for r := res; r != nil; r = r.ai_next {
		// We only asked for SOCK_STREAM, but check anyhow.
		if r.ai_socktype != C.SOCK_STREAM {
			continue
		}
		switch r.ai_family {
		default:
			continue
		case C.AF_INET:
			sa := (*syscall.RawSockaddrInet4)(unsafe.Pointer(r.ai_addr))
			addrs = append(addrs, copyIP(sa.Addr[:]))
		case C.AF_INET6:
			sa := (*syscall.RawSockaddrInet6)(unsafe.Pointer(r.ai_addr))
			addrs = append(addrs, copyIP(sa.Addr[:]))
		}
	}
	return addrs, cname, nil, true
}

func cgoLookupIP(name string) (addrs []IP, err error, completed bool) {
	addrs, _, err, completed = cgoLookupIPCNAME(name)
	return
}

func cgoLookupCNAME(name string) (cname string, err error, completed bool) {
	_, cname, err, completed = cgoLookupIPCNAME(name)
	return
}

func copyIP(x IP) IP {
	if len(x) < 16 {
		return x.To16()
	}
	y := make(IP, len(x))
	copy(y, x)
	return y
}

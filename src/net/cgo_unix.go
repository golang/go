// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cgo,!netgo
// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris

package net

/*
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <unistd.h>
#include <string.h>

// If nothing else defined EAI_OVERFLOW, make sure it has a value.
#ifndef EAI_OVERFLOW
#define EAI_OVERFLOW -12
#endif
*/
import "C"

import (
	"context"
	"os"
	"syscall"
	"unsafe"
)

// An addrinfoErrno represents a getaddrinfo, getnameinfo-specific
// error number. It's a signed number and a zero value is a non-error
// by convention.
type addrinfoErrno int

func (eai addrinfoErrno) Error() string   { return C.GoString(C.gai_strerror(C.int(eai))) }
func (eai addrinfoErrno) Temporary() bool { return eai == C.EAI_AGAIN }
func (eai addrinfoErrno) Timeout() bool   { return false }

func (eai addrinfoErrno) Is(target error) bool {
	switch target {
	case os.ErrTemporary:
		return eai.Temporary()
	case os.ErrTimeout:
		return eai.Timeout()
	}
	return false
}

type portLookupResult struct {
	port int
	err  error
}

type ipLookupResult struct {
	addrs []IPAddr
	cname string
	err   error
}

type reverseLookupResult struct {
	names []string
	err   error
}

func cgoLookupHost(ctx context.Context, name string) (hosts []string, err error, completed bool) {
	addrs, err, completed := cgoLookupIP(ctx, "ip", name)
	for _, addr := range addrs {
		hosts = append(hosts, addr.String())
	}
	return
}

func cgoLookupPort(ctx context.Context, network, service string) (port int, err error, completed bool) {
	var hints C.struct_addrinfo
	switch network {
	case "": // no hints
	case "tcp", "tcp4", "tcp6":
		hints.ai_socktype = C.SOCK_STREAM
		hints.ai_protocol = C.IPPROTO_TCP
	case "udp", "udp4", "udp6":
		hints.ai_socktype = C.SOCK_DGRAM
		hints.ai_protocol = C.IPPROTO_UDP
	default:
		return 0, &DNSError{Err: "unknown network", Name: network + "/" + service}, true
	}
	switch ipVersion(network) {
	case '4':
		hints.ai_family = C.AF_INET
	case '6':
		hints.ai_family = C.AF_INET6
	}
	if ctx.Done() == nil {
		port, err := cgoLookupServicePort(&hints, network, service)
		return port, err, true
	}
	result := make(chan portLookupResult, 1)
	go cgoPortLookup(result, &hints, network, service)
	select {
	case r := <-result:
		return r.port, r.err, true
	case <-ctx.Done():
		// Since there isn't a portable way to cancel the lookup,
		// we just let it finish and write to the buffered channel.
		return 0, mapErr(ctx.Err()), false
	}
}

func cgoLookupServicePort(hints *C.struct_addrinfo, network, service string) (port int, err error) {
	cservice := make([]byte, len(service)+1)
	copy(cservice, service)
	// Lowercase the C service name.
	for i, b := range cservice[:len(service)] {
		cservice[i] = lowerASCII(b)
	}
	var res *C.struct_addrinfo
	gerrno, err := C.getaddrinfo(nil, (*C.char)(unsafe.Pointer(&cservice[0])), hints, &res)
	if gerrno != 0 {
		isTemporary := false
		switch gerrno {
		case C.EAI_SYSTEM:
			if err == nil { // see golang.org/issue/6232
				err = syscall.EMFILE
			}
		default:
			err = addrinfoErrno(gerrno)
			isTemporary = addrinfoErrno(gerrno).Temporary()
		}
		return 0, &DNSError{Err: err.Error(), Name: network + "/" + service, IsTemporary: isTemporary}
	}
	defer C.freeaddrinfo(res)

	for r := res; r != nil; r = r.ai_next {
		switch r.ai_family {
		case C.AF_INET:
			sa := (*syscall.RawSockaddrInet4)(unsafe.Pointer(r.ai_addr))
			p := (*[2]byte)(unsafe.Pointer(&sa.Port))
			return int(p[0])<<8 | int(p[1]), nil
		case C.AF_INET6:
			sa := (*syscall.RawSockaddrInet6)(unsafe.Pointer(r.ai_addr))
			p := (*[2]byte)(unsafe.Pointer(&sa.Port))
			return int(p[0])<<8 | int(p[1]), nil
		}
	}
	return 0, &DNSError{Err: "unknown port", Name: network + "/" + service}
}

func cgoPortLookup(result chan<- portLookupResult, hints *C.struct_addrinfo, network, service string) {
	port, err := cgoLookupServicePort(hints, network, service)
	result <- portLookupResult{port, err}
}

func cgoLookupIPCNAME(network, name string) (addrs []IPAddr, cname string, err error) {
	acquireThread()
	defer releaseThread()

	var hints C.struct_addrinfo
	hints.ai_flags = cgoAddrInfoFlags
	hints.ai_socktype = C.SOCK_STREAM
	hints.ai_family = C.AF_UNSPEC
	switch ipVersion(network) {
	case '4':
		hints.ai_family = C.AF_INET
	case '6':
		hints.ai_family = C.AF_INET6
	}

	h := make([]byte, len(name)+1)
	copy(h, name)
	var res *C.struct_addrinfo
	gerrno, err := C.getaddrinfo((*C.char)(unsafe.Pointer(&h[0])), nil, &hints, &res)
	if gerrno != 0 {
		isErrorNoSuchHost := false
		isTemporary := false
		switch gerrno {
		case C.EAI_SYSTEM:
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
		case C.EAI_NONAME:
			err = errNoSuchHost
			isErrorNoSuchHost = true
		default:
			err = addrinfoErrno(gerrno)
			isTemporary = addrinfoErrno(gerrno).Temporary()
		}

		return nil, "", &DNSError{Err: err.Error(), Name: name, IsNotFound: isErrorNoSuchHost, IsTemporary: isTemporary}
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
		case C.AF_INET:
			sa := (*syscall.RawSockaddrInet4)(unsafe.Pointer(r.ai_addr))
			addr := IPAddr{IP: copyIP(sa.Addr[:])}
			addrs = append(addrs, addr)
		case C.AF_INET6:
			sa := (*syscall.RawSockaddrInet6)(unsafe.Pointer(r.ai_addr))
			addr := IPAddr{IP: copyIP(sa.Addr[:]), Zone: zoneCache.name(int(sa.Scope_id))}
			addrs = append(addrs, addr)
		}
	}
	return addrs, cname, nil
}

func cgoIPLookup(result chan<- ipLookupResult, network, name string) {
	addrs, cname, err := cgoLookupIPCNAME(network, name)
	result <- ipLookupResult{addrs, cname, err}
}

func cgoLookupIP(ctx context.Context, network, name string) (addrs []IPAddr, err error, completed bool) {
	if ctx.Done() == nil {
		addrs, _, err = cgoLookupIPCNAME(network, name)
		return addrs, err, true
	}
	result := make(chan ipLookupResult, 1)
	go cgoIPLookup(result, network, name)
	select {
	case r := <-result:
		return r.addrs, r.err, true
	case <-ctx.Done():
		return nil, mapErr(ctx.Err()), false
	}
}

func cgoLookupCNAME(ctx context.Context, name string) (cname string, err error, completed bool) {
	if ctx.Done() == nil {
		_, cname, err = cgoLookupIPCNAME("ip", name)
		return cname, err, true
	}
	result := make(chan ipLookupResult, 1)
	go cgoIPLookup(result, "ip", name)
	select {
	case r := <-result:
		return r.cname, r.err, true
	case <-ctx.Done():
		return "", mapErr(ctx.Err()), false
	}
}

// These are roughly enough for the following:
//
// Source		Encoding			Maximum length of single name entry
// Unicast DNS		ASCII or			<=253 + a NUL terminator
//			Unicode in RFC 5892		252 * total number of labels + delimiters + a NUL terminator
// Multicast DNS	UTF-8 in RFC 5198 or		<=253 + a NUL terminator
//			the same as unicast DNS ASCII	<=253 + a NUL terminator
// Local database	various				depends on implementation
const (
	nameinfoLen    = 64
	maxNameinfoLen = 4096
)

func cgoLookupPTR(ctx context.Context, addr string) (names []string, err error, completed bool) {
	var zone string
	ip := parseIPv4(addr)
	if ip == nil {
		ip, zone = parseIPv6Zone(addr)
	}
	if ip == nil {
		return nil, &DNSError{Err: "invalid address", Name: addr}, true
	}
	sa, salen := cgoSockaddr(ip, zone)
	if sa == nil {
		return nil, &DNSError{Err: "invalid address " + ip.String(), Name: addr}, true
	}
	if ctx.Done() == nil {
		names, err := cgoLookupAddrPTR(addr, sa, salen)
		return names, err, true
	}
	result := make(chan reverseLookupResult, 1)
	go cgoReverseLookup(result, addr, sa, salen)
	select {
	case r := <-result:
		return r.names, r.err, true
	case <-ctx.Done():
		return nil, mapErr(ctx.Err()), false
	}
}

func cgoLookupAddrPTR(addr string, sa *C.struct_sockaddr, salen C.socklen_t) (names []string, err error) {
	acquireThread()
	defer releaseThread()

	var gerrno int
	var b []byte
	for l := nameinfoLen; l <= maxNameinfoLen; l *= 2 {
		b = make([]byte, l)
		gerrno, err = cgoNameinfoPTR(b, sa, salen)
		if gerrno == 0 || gerrno != C.EAI_OVERFLOW {
			break
		}
	}
	if gerrno != 0 {
		isTemporary := false
		switch gerrno {
		case C.EAI_SYSTEM:
			if err == nil { // see golang.org/issue/6232
				err = syscall.EMFILE
			}
		default:
			err = addrinfoErrno(gerrno)
			isTemporary = addrinfoErrno(gerrno).Temporary()
		}
		return nil, &DNSError{Err: err.Error(), Name: addr, IsTemporary: isTemporary}
	}
	for i := 0; i < len(b); i++ {
		if b[i] == 0 {
			b = b[:i]
			break
		}
	}
	return []string{absDomainName(b)}, nil
}

func cgoReverseLookup(result chan<- reverseLookupResult, addr string, sa *C.struct_sockaddr, salen C.socklen_t) {
	names, err := cgoLookupAddrPTR(addr, sa, salen)
	result <- reverseLookupResult{names, err}
}

func cgoSockaddr(ip IP, zone string) (*C.struct_sockaddr, C.socklen_t) {
	if ip4 := ip.To4(); ip4 != nil {
		return cgoSockaddrInet4(ip4), C.socklen_t(syscall.SizeofSockaddrInet4)
	}
	if ip6 := ip.To16(); ip6 != nil {
		return cgoSockaddrInet6(ip6, zoneCache.index(zone)), C.socklen_t(syscall.SizeofSockaddrInet6)
	}
	return nil, 0
}

func copyIP(x IP) IP {
	if len(x) < 16 {
		return x.To16()
	}
	y := make(IP, len(x))
	copy(y, x)
	return y
}

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file is intended to be used in non-cgo builds of darwin binaries,
// in particular when cross-compiling a darwin binary from a non-darwin machine.
// All OS calls on darwin have to be done via C libraries, and this code makes such
// calls with the runtime's help. (It is a C call but does not require the cgo tool to
// be compiled, and as such it is possible to build even when cross-compiling.)
//
// The specific C library calls are to res_init and res_search from /usr/lib/system/libsystem_info.dylib.
// Unfortunately, an ordinary C program calling these names would actually end up with
// res_9_init and res_9_search from /usr/lib/libresolv.dylib, not libsystem_info.
// It may well be that the libsystem_info routines are completely unused on macOS systems
// except for this code. At the least, they have the following problems:
//
//	- TypeALL requests do not work, so if we want both IPv4 and IPv6 addresses,
//	  we have to do two requests, one for TypeA and one for TypeAAAA.
//	- TypeCNAME requests hang indefinitely.
//	- TypePTR requests fail unconditionally.
//	- Detailed error information is stored in the global h_errno value,
//	  which cannot be accessed safely (it is not per-thread like errno).
//	- The routines may not be safe to call from multiple threads.
//	  If you run net.test under lldb, that emits syslog prints to stderr
//	  that suggest double-free problems. (If not running under lldb,
//	  it is unclear where the syslog prints go, if anywhere.)
//
// This code is marked for deletion. If it is to be revived, it should be changed to use
// res_9_init and res_9_search from libresolv and special care should be paid to
// error detail and thread safety.

// +build !netgo,!cgo
// +build darwin

package net

import (
	"context"
	"errors"
	"sync"

	"golang.org/x/net/dns/dnsmessage"
)

type addrinfoErrno int

func (eai addrinfoErrno) Error() string   { return "<nil>" }
func (eai addrinfoErrno) Temporary() bool { return false }
func (eai addrinfoErrno) Timeout() bool   { return false }

func cgoLookupHost(ctx context.Context, name string) (addrs []string, err error, completed bool) {
	// The 4-suffix indicates IPv4, TypeA lookups.
	// The 6-suffix indicates IPv6, TypeAAAA lookups.
	// If resSearch is updated to call the libresolv res_9_search (see comment at top of file),
	// it may be possible to make one call for TypeALL
	// and get both address kinds out.
	r4, err4 := resSearch(ctx, name, int32(dnsmessage.TypeA), int32(dnsmessage.ClassINET))
	if err4 == nil {
		addrs, err4 = appendHostsFromResources(addrs, r4)
	}
	r6, err6 := resSearch(ctx, name, int32(dnsmessage.TypeAAAA), int32(dnsmessage.ClassINET))
	if err6 == nil {
		addrs, err6 = appendHostsFromResources(addrs, r6)
	}
	if err4 != nil && err6 != nil {
		return nil, err4, false
	}
	return addrs, nil, true
}

func cgoLookupPort(ctx context.Context, network, service string) (port int, err error, completed bool) {
	port, err = goLookupPort(network, service) // we can just use netgo lookup
	return port, err, err == nil
}

func cgoLookupIP(ctx context.Context, network, name string) (addrs []IPAddr, err error, completed bool) {
	// The 4-suffix indicates IPv4, TypeA lookups.
	// The 6-suffix indicates IPv6, TypeAAAA lookups.
	// If resSearch is updated to call the libresolv res_9_search (see comment at top of file),
	// it may be possible to make one call for TypeALL (when vers != '6' and vers != '4')
	// and get both address kinds out.
	var r4, r6 []dnsmessage.Resource
	var err4, err6 error
	vers := ipVersion(network)
	if vers != '6' {
		r4, err4 = resSearch(ctx, name, int32(dnsmessage.TypeA), int32(dnsmessage.ClassINET))
		if err4 == nil {
			addrs, err4 = appendIPsFromResources(addrs, r4)
		}
	}
	if vers != '4' {
		r6, err6 = resSearch(ctx, name, int32(dnsmessage.TypeAAAA), int32(dnsmessage.ClassINET))
		if err6 == nil {
			addrs, err6 = appendIPsFromResources(addrs, r6)
		}
	}
	if err4 != nil && err6 != nil {
		return nil, err4, false
	}

	return addrs, nil, true
}

func cgoLookupCNAME(ctx context.Context, name string) (cname string, err error, completed bool) {
	resources, err := resSearch(ctx, name, int32(dnsmessage.TypeCNAME), int32(dnsmessage.ClassINET))
	if err != nil {
		return
	}
	cname, err = parseCNAMEFromResources(resources)
	if err != nil {
		return "", err, false
	}
	return cname, nil, true
}

func cgoLookupPTR(ctx context.Context, addr string) (ptrs []string, err error, completed bool) {
	resources, err := resSearch(ctx, addr, int32(dnsmessage.TypePTR), int32(dnsmessage.ClassINET))
	if err != nil {
		return
	}
	ptrs, err = parsePTRsFromResources(resources)
	if err != nil {
		return
	}
	return ptrs, nil, true
}

var (
	resInitOnce   sync.Once
	resInitResult int32
)

// resSearch will make a call to the 'res_search' routine in libSystem
// and parse the output as a slice of resource resources which can then be parsed
func resSearch(ctx context.Context, hostname string, rtype, class int32) ([]dnsmessage.Resource, error) {
	// We have to use res_init and res_search, but these do not set errno on failure.
	// (They set h_errno, which is a global int shared by all threads and therefore
	// racy to use.)
	// https://opensource.apple.com/source/Libinfo/Libinfo-517.200.9/dns.subproj/res_query.c.auto.html
	resInitOnce.Do(func() {
		resInitResult = res_init()
	})
	if resInitResult < 0 {
		return nil, errors.New("res_init failure")
	}

	// res_search does not set errno.
	// It returns the size of the DNS response packet.
	// But if the DNS response packet contains failure-like response codes,
	// res_search returns -1 even though it has copied the packet into buf,
	// giving us no way to find out how big the packet is.
	// For now, we are willing to take res_search's word that there's nothing
	// useful in the response, even though there *is* a response.
	name := make([]byte, len(hostname)+1) // +1 for NUL at end for C
	copy(name, hostname)
	var buf [1024]byte
	size, _ := res_search(&name[0], class, rtype, &buf[0], int32(len(buf)))
	if size <= 0 {
		return nil, errors.New("res_search failure")
	}

	var p dnsmessage.Parser
	if _, err := p.Start(buf[:size]); err != nil {
		return nil, err
	}
	p.SkipAllQuestions()
	resources, err := p.AllAnswers()
	if err != nil {
		return nil, err
	}
	return resources, nil
}

func copyBytes(x []byte) []byte {
	y := make([]byte, len(x))
	copy(y, x)
	return y
}

func appendHostsFromResources(answers []string, resources []dnsmessage.Resource) ([]string, error) {
	for i := range resources {
		switch resources[i].Header.Type {
		case dnsmessage.TypeA:
			b := resources[i].Body.(*dnsmessage.AResource)
			answers = append(answers, IP(b.A[:]).String())
		case dnsmessage.TypeAAAA:
			b := resources[i].Body.(*dnsmessage.AAAAResource)
			answers = append(answers, IP(b.AAAA[:]).String())
		default:
			return nil, errors.New("could not parse an A or AAAA response from message buffer")
		}
	}
	return answers, nil
}

func appendIPsFromResources(answers []IPAddr, resources []dnsmessage.Resource) ([]IPAddr, error) {
	for i := range resources {
		switch resources[i].Header.Type {
		case dnsmessage.TypeA:
			b := resources[i].Body.(*dnsmessage.AResource)
			answers = append(answers, IPAddr{IP: IP(copyBytes(b.A[:]))})
		case dnsmessage.TypeAAAA:
			b := resources[i].Body.(*dnsmessage.AAAAResource)
			answers = append(answers, IPAddr{IP: IP(copyBytes(b.AAAA[:]))})
		default:
			return nil, errors.New("could not parse an A or AAAA response from message buffer")
		}
	}
	return answers, nil
}

func parseCNAMEFromResources(resources []dnsmessage.Resource) (string, error) {
	if len(resources) == 0 {
		return "", errors.New("no CNAME record received")
	}
	c, ok := resources[0].Body.(*dnsmessage.CNAMEResource)
	if !ok {
		return "", errors.New("could not parse CNAME record")
	}
	return c.CNAME.String(), nil
}

func parsePTRsFromResources(resources []dnsmessage.Resource) ([]string, error) {
	var answers []string
	for i := range resources {
		switch resources[i].Header.Type {
		case dnsmessage.TypePTR:
			p := resources[0].Body.(*dnsmessage.PTRResource)
			answers = append(answers, p.PTR.String())
		default:
			return nil, errors.New("could not parse a PTR response from message buffer")

		}
	}
	return answers, nil
}

// res_init and res_search are defined in runtime/lookup_darwin.go

func res_init() int32

func res_search(dname *byte, class int32, rtype int32, answer *byte, anslen int32) (int32, int32)

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"syscall"
	"unsafe"
	"os"
	"sync"
)

var hostentLock sync.Mutex
var serventLock sync.Mutex

func goLookupHost(name string) (addrs []string, err os.Error) {
	ips, err := goLookupIP(name)
	if err != nil {
		return
	}
	addrs = make([]string, 0, len(ips))
	for _, ip := range ips {
		addrs = append(addrs, ip.String())
	}
	return
}

func goLookupIP(name string) (addrs []IP, err os.Error) {
	hostentLock.Lock()
	defer hostentLock.Unlock()
	h, e := syscall.GetHostByName(name)
	if e != 0 {
		return nil, os.NewSyscallError("GetHostByName", e)
	}
	switch h.AddrType {
	case syscall.AF_INET:
		i := 0
		addrs = make([]IP, 100) // plenty of room to grow
		for p := (*[100](*[4]byte))(unsafe.Pointer(h.AddrList)); i < cap(addrs) && p[i] != nil; i++ {
			addrs[i] = IPv4(p[i][0], p[i][1], p[i][2], p[i][3])
		}
		addrs = addrs[0:i]
	default: // TODO(vcc): Implement non IPv4 address lookups.
		return nil, os.NewSyscallError("LookupHost", syscall.EWINDOWS)
	}
	return addrs, nil
}

func goLookupCNAME(name string) (cname string, err os.Error) {
	var r *syscall.DNSRecord
	e := syscall.DnsQuery(name, syscall.DNS_TYPE_CNAME, 0, nil, &r, nil)
	if int(e) != 0 {
		return "", os.NewSyscallError("LookupCNAME", int(e))
	}
	defer syscall.DnsRecordListFree(r, 1)
	if r != nil && r.Type == syscall.DNS_TYPE_CNAME {
		v := (*syscall.DNSPTRData)(unsafe.Pointer(&r.Data[0]))
		cname = syscall.UTF16ToString((*[256]uint16)(unsafe.Pointer(v.Host))[:]) + "."
	}
	return
}

type SRV struct {
	Target   string
	Port     uint16
	Priority uint16
	Weight   uint16
}

func LookupSRV(service, proto, name string) (cname string, addrs []*SRV, err os.Error) {
	var r *syscall.DNSRecord
	target := "_" + service + "._" + proto + "." + name
	e := syscall.DnsQuery(target, syscall.DNS_TYPE_SRV, 0, nil, &r, nil)
	if int(e) != 0 {
		return "", nil, os.NewSyscallError("LookupSRV", int(e))
	}
	defer syscall.DnsRecordListFree(r, 1)
	addrs = make([]*SRV, 100)
	i := 0
	for p := r; p != nil && p.Type == syscall.DNS_TYPE_SRV; p = p.Next {
		v := (*syscall.DNSSRVData)(unsafe.Pointer(&p.Data[0]))
		addrs[i] = &SRV{syscall.UTF16ToString((*[256]uint16)(unsafe.Pointer(v.Target))[:]), v.Port, v.Priority, v.Weight}
		i++
	}
	addrs = addrs[0:i]
	return name, addrs, nil
}

func goLookupPort(network, service string) (port int, err os.Error) {
	switch network {
	case "tcp4", "tcp6":
		network = "tcp"
	case "udp4", "udp6":
		network = "udp"
	}
	serventLock.Lock()
	defer serventLock.Unlock()
	s, e := syscall.GetServByName(service, network)
	if e != 0 {
		return 0, os.NewSyscallError("GetServByName", e)
	}
	return int(syscall.Ntohs(s.Port)), nil
}

// TODO(brainman): Following code is only to get tests running.

func isDomainName(s string) bool {
	panic("unimplemented")
}

func reverseaddr(addr string) (arpa string, err os.Error) {
	panic("unimplemented")
}

func answer(name, server string, dns *dnsMsg, qtype uint16) (cname string, addrs []dnsRR, err os.Error) {
	panic("unimplemented")
}

// DNSError represents a DNS lookup error.
type DNSError struct {
	Error     string // description of the error
	Name      string // name looked for
	Server    string // server used
	IsTimeout bool
}

func (e *DNSError) String() string {
	if e == nil {
		return "<nil>"
	}
	s := "lookup " + e.Name
	if e.Server != "" {
		s += " on " + e.Server
	}
	s += ": " + e.Error
	return s
}

func (e *DNSError) Timeout() bool   { return e.IsTimeout }
func (e *DNSError) Temporary() bool { return e.IsTimeout }

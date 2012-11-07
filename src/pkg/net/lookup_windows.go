// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"os"
	"sync"
	"syscall"
	"unsafe"
)

var (
	protoentLock sync.Mutex
	hostentLock  sync.Mutex
	serventLock  sync.Mutex
)

// lookupProtocol looks up IP protocol name and returns correspondent protocol number.
func lookupProtocol(name string) (proto int, err error) {
	protoentLock.Lock()
	defer protoentLock.Unlock()
	p, err := syscall.GetProtoByName(name)
	if err != nil {
		return 0, os.NewSyscallError("GetProtoByName", err)
	}
	return int(p.Proto), nil
}

func lookupHost(name string) (addrs []string, err error) {
	ips, err := LookupIP(name)
	if err != nil {
		return
	}
	addrs = make([]string, 0, len(ips))
	for _, ip := range ips {
		addrs = append(addrs, ip.String())
	}
	return
}

var lookupIP = oldLookupIP

func oldLookupIP(name string) (addrs []IP, err error) {
	hostentLock.Lock()
	defer hostentLock.Unlock()
	h, err := syscall.GetHostByName(name)
	if err != nil {
		return nil, os.NewSyscallError("GetHostByName", err)
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
		return nil, os.NewSyscallError("LookupIP", syscall.EWINDOWS)
	}
	return addrs, nil
}

func newLookupIP(name string) (addrs []IP, err error) {
	hints := syscall.AddrinfoW{
		Family:   syscall.AF_UNSPEC,
		Socktype: syscall.SOCK_STREAM,
		Protocol: syscall.IPPROTO_IP,
	}
	var result *syscall.AddrinfoW
	e := syscall.GetAddrInfoW(syscall.StringToUTF16Ptr(name), nil, &hints, &result)
	if e != nil {
		return nil, os.NewSyscallError("GetAddrInfoW", e)
	}
	defer syscall.FreeAddrInfoW(result)
	addrs = make([]IP, 0, 5)
	for ; result != nil; result = result.Next {
		addr := unsafe.Pointer(result.Addr)
		switch result.Family {
		case syscall.AF_INET:
			a := (*syscall.RawSockaddrInet4)(addr).Addr
			addrs = append(addrs, IPv4(a[0], a[1], a[2], a[3]))
		case syscall.AF_INET6:
			a := (*syscall.RawSockaddrInet6)(addr).Addr
			addrs = append(addrs, IP{a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]})
		default:
			return nil, os.NewSyscallError("LookupIP", syscall.EWINDOWS)
		}
	}
	return addrs, nil
}

func lookupPort(network, service string) (port int, err error) {
	switch network {
	case "tcp4", "tcp6":
		network = "tcp"
	case "udp4", "udp6":
		network = "udp"
	}
	serventLock.Lock()
	defer serventLock.Unlock()
	s, err := syscall.GetServByName(service, network)
	if err != nil {
		return 0, os.NewSyscallError("GetServByName", err)
	}
	return int(syscall.Ntohs(s.Port)), nil
}

func lookupCNAME(name string) (cname string, err error) {
	var r *syscall.DNSRecord
	e := syscall.DnsQuery(name, syscall.DNS_TYPE_CNAME, 0, nil, &r, nil)
	if e != nil {
		return "", os.NewSyscallError("LookupCNAME", e)
	}
	defer syscall.DnsRecordListFree(r, 1)
	if r != nil && r.Type == syscall.DNS_TYPE_CNAME {
		v := (*syscall.DNSPTRData)(unsafe.Pointer(&r.Data[0]))
		cname = syscall.UTF16ToString((*[256]uint16)(unsafe.Pointer(v.Host))[:]) + "."
	}
	return
}

func lookupSRV(service, proto, name string) (cname string, addrs []*SRV, err error) {
	var target string
	if service == "" && proto == "" {
		target = name
	} else {
		target = "_" + service + "._" + proto + "." + name
	}
	var r *syscall.DNSRecord
	e := syscall.DnsQuery(target, syscall.DNS_TYPE_SRV, 0, nil, &r, nil)
	if e != nil {
		return "", nil, os.NewSyscallError("LookupSRV", e)
	}
	defer syscall.DnsRecordListFree(r, 1)
	addrs = make([]*SRV, 0, 10)
	for p := r; p != nil && p.Type == syscall.DNS_TYPE_SRV; p = p.Next {
		v := (*syscall.DNSSRVData)(unsafe.Pointer(&p.Data[0]))
		addrs = append(addrs, &SRV{syscall.UTF16ToString((*[256]uint16)(unsafe.Pointer(v.Target))[:]), v.Port, v.Priority, v.Weight})
	}
	byPriorityWeight(addrs).sort()
	return name, addrs, nil
}

func lookupMX(name string) (mx []*MX, err error) {
	var r *syscall.DNSRecord
	e := syscall.DnsQuery(name, syscall.DNS_TYPE_MX, 0, nil, &r, nil)
	if e != nil {
		return nil, os.NewSyscallError("LookupMX", e)
	}
	defer syscall.DnsRecordListFree(r, 1)
	mx = make([]*MX, 0, 10)
	for p := r; p != nil && p.Type == syscall.DNS_TYPE_MX; p = p.Next {
		v := (*syscall.DNSMXData)(unsafe.Pointer(&p.Data[0]))
		mx = append(mx, &MX{syscall.UTF16ToString((*[256]uint16)(unsafe.Pointer(v.NameExchange))[:]) + ".", v.Preference})
	}
	byPref(mx).sort()
	return mx, nil
}

func lookupNS(name string) (ns []*NS, err error) {
	var r *syscall.DNSRecord
	e := syscall.DnsQuery(name, syscall.DNS_TYPE_NS, 0, nil, &r, nil)
	if e != nil {
		return nil, os.NewSyscallError("LookupNS", e)
	}
	defer syscall.DnsRecordListFree(r, 1)
	ns = make([]*NS, 0, 10)
	for p := r; p != nil && p.Type == syscall.DNS_TYPE_NS; p = p.Next {
		v := (*syscall.DNSPTRData)(unsafe.Pointer(&p.Data[0]))
		ns = append(ns, &NS{syscall.UTF16ToString((*[256]uint16)(unsafe.Pointer(v.Host))[:]) + "."})
	}
	return ns, nil
}

func lookupTXT(name string) (txt []string, err error) {
	var r *syscall.DNSRecord
	e := syscall.DnsQuery(name, syscall.DNS_TYPE_TEXT, 0, nil, &r, nil)
	if e != nil {
		return nil, os.NewSyscallError("LookupTXT", e)
	}
	defer syscall.DnsRecordListFree(r, 1)
	txt = make([]string, 0, 10)
	if r != nil && r.Type == syscall.DNS_TYPE_TEXT {
		d := (*syscall.DNSTXTData)(unsafe.Pointer(&r.Data[0]))
		for _, v := range (*[1 << 10]*uint16)(unsafe.Pointer(&(d.StringArray[0])))[:d.StringCount] {
			s := syscall.UTF16ToString((*[1 << 20]uint16)(unsafe.Pointer(v))[:])
			txt = append(txt, s)
		}
	}
	return
}

func lookupAddr(addr string) (name []string, err error) {
	arpa, err := reverseaddr(addr)
	if err != nil {
		return nil, err
	}
	var r *syscall.DNSRecord
	e := syscall.DnsQuery(arpa, syscall.DNS_TYPE_PTR, 0, nil, &r, nil)
	if e != nil {
		return nil, os.NewSyscallError("LookupAddr", e)
	}
	defer syscall.DnsRecordListFree(r, 1)
	name = make([]string, 0, 10)
	for p := r; p != nil && p.Type == syscall.DNS_TYPE_PTR; p = p.Next {
		v := (*syscall.DNSPTRData)(unsafe.Pointer(&p.Data[0]))
		name = append(name, syscall.UTF16ToString((*[256]uint16)(unsafe.Pointer(v.Host))[:]))
	}
	return name, nil
}

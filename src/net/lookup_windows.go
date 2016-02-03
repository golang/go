// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"os"
	"runtime"
	"syscall"
	"unsafe"
)

var (
	lookupPort = oldLookupPort
	lookupIP   = oldLookupIP
)

func getprotobyname(name string) (proto int, err error) {
	p, err := syscall.GetProtoByName(name)
	if err != nil {
		return 0, os.NewSyscallError("getprotobyname", err)
	}
	return int(p.Proto), nil
}

// lookupProtocol looks up IP protocol name and returns correspondent protocol number.
func lookupProtocol(name string) (int, error) {
	// GetProtoByName return value is stored in thread local storage.
	// Start new os thread before the call to prevent races.
	type result struct {
		proto int
		err   error
	}
	ch := make(chan result)
	go func() {
		acquireThread()
		defer releaseThread()
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		proto, err := getprotobyname(name)
		ch <- result{proto: proto, err: err}
	}()
	r := <-ch
	if r.err != nil {
		if proto, ok := protocols[name]; ok {
			return proto, nil
		}
		r.err = &DNSError{Err: r.err.Error(), Name: name}
	}
	return r.proto, r.err
}

func lookupHost(name string) ([]string, error) {
	ips, err := LookupIP(name)
	if err != nil {
		return nil, err
	}
	addrs := make([]string, 0, len(ips))
	for _, ip := range ips {
		addrs = append(addrs, ip.String())
	}
	return addrs, nil
}

func gethostbyname(name string) (addrs []IPAddr, err error) {
	// caller already acquired thread
	h, err := syscall.GetHostByName(name)
	if err != nil {
		return nil, os.NewSyscallError("gethostbyname", err)
	}
	switch h.AddrType {
	case syscall.AF_INET:
		i := 0
		addrs = make([]IPAddr, 100) // plenty of room to grow
		for p := (*[100](*[4]byte))(unsafe.Pointer(h.AddrList)); i < cap(addrs) && p[i] != nil; i++ {
			addrs[i] = IPAddr{IP: IPv4(p[i][0], p[i][1], p[i][2], p[i][3])}
		}
		addrs = addrs[0:i]
	default: // TODO(vcc): Implement non IPv4 address lookups.
		return nil, syscall.EWINDOWS
	}
	return addrs, nil
}

func oldLookupIP(name string) ([]IPAddr, error) {
	// GetHostByName return value is stored in thread local storage.
	// Start new os thread before the call to prevent races.
	type result struct {
		addrs []IPAddr
		err   error
	}
	ch := make(chan result)
	go func() {
		acquireThread()
		defer releaseThread()
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		addrs, err := gethostbyname(name)
		ch <- result{addrs: addrs, err: err}
	}()
	r := <-ch
	if r.err != nil {
		r.err = &DNSError{Err: r.err.Error(), Name: name}
	}
	return r.addrs, r.err
}

func newLookupIP(name string) ([]IPAddr, error) {
	acquireThread()
	defer releaseThread()
	hints := syscall.AddrinfoW{
		Family:   syscall.AF_UNSPEC,
		Socktype: syscall.SOCK_STREAM,
		Protocol: syscall.IPPROTO_IP,
	}
	var result *syscall.AddrinfoW
	e := syscall.GetAddrInfoW(syscall.StringToUTF16Ptr(name), nil, &hints, &result)
	if e != nil {
		return nil, &DNSError{Err: os.NewSyscallError("getaddrinfow", e).Error(), Name: name}
	}
	defer syscall.FreeAddrInfoW(result)
	addrs := make([]IPAddr, 0, 5)
	for ; result != nil; result = result.Next {
		addr := unsafe.Pointer(result.Addr)
		switch result.Family {
		case syscall.AF_INET:
			a := (*syscall.RawSockaddrInet4)(addr).Addr
			addrs = append(addrs, IPAddr{IP: IPv4(a[0], a[1], a[2], a[3])})
		case syscall.AF_INET6:
			a := (*syscall.RawSockaddrInet6)(addr).Addr
			zone := zoneToString(int((*syscall.RawSockaddrInet6)(addr).Scope_id))
			addrs = append(addrs, IPAddr{IP: IP{a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]}, Zone: zone})
		default:
			return nil, &DNSError{Err: syscall.EWINDOWS.Error(), Name: name}
		}
	}
	return addrs, nil
}

func getservbyname(network, service string) (int, error) {
	acquireThread()
	defer releaseThread()
	switch network {
	case "tcp4", "tcp6":
		network = "tcp"
	case "udp4", "udp6":
		network = "udp"
	}
	s, err := syscall.GetServByName(service, network)
	if err != nil {
		return 0, os.NewSyscallError("getservbyname", err)
	}
	return int(syscall.Ntohs(s.Port)), nil
}

func oldLookupPort(network, service string) (int, error) {
	// GetServByName return value is stored in thread local storage.
	// Start new os thread before the call to prevent races.
	type result struct {
		port int
		err  error
	}
	ch := make(chan result)
	go func() {
		acquireThread()
		defer releaseThread()
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		port, err := getservbyname(network, service)
		ch <- result{port: port, err: err}
	}()
	r := <-ch
	if r.err != nil {
		r.err = &DNSError{Err: r.err.Error(), Name: network + "/" + service}
	}
	return r.port, r.err
}

func newLookupPort(network, service string) (int, error) {
	acquireThread()
	defer releaseThread()
	var stype int32
	switch network {
	case "tcp4", "tcp6":
		stype = syscall.SOCK_STREAM
	case "udp4", "udp6":
		stype = syscall.SOCK_DGRAM
	}
	hints := syscall.AddrinfoW{
		Family:   syscall.AF_UNSPEC,
		Socktype: stype,
		Protocol: syscall.IPPROTO_IP,
	}
	var result *syscall.AddrinfoW
	e := syscall.GetAddrInfoW(nil, syscall.StringToUTF16Ptr(service), &hints, &result)
	if e != nil {
		return 0, &DNSError{Err: os.NewSyscallError("getaddrinfow", e).Error(), Name: network + "/" + service}
	}
	defer syscall.FreeAddrInfoW(result)
	if result == nil {
		return 0, &DNSError{Err: syscall.EINVAL.Error(), Name: network + "/" + service}
	}
	addr := unsafe.Pointer(result.Addr)
	switch result.Family {
	case syscall.AF_INET:
		a := (*syscall.RawSockaddrInet4)(addr)
		return int(syscall.Ntohs(a.Port)), nil
	case syscall.AF_INET6:
		a := (*syscall.RawSockaddrInet6)(addr)
		return int(syscall.Ntohs(a.Port)), nil
	}
	return 0, &DNSError{Err: syscall.EINVAL.Error(), Name: network + "/" + service}
}

func lookupCNAME(name string) (string, error) {
	acquireThread()
	defer releaseThread()
	var r *syscall.DNSRecord
	e := syscall.DnsQuery(name, syscall.DNS_TYPE_CNAME, 0, nil, &r, nil)
	// windows returns DNS_INFO_NO_RECORDS if there are no CNAME-s
	if errno, ok := e.(syscall.Errno); ok && errno == syscall.DNS_INFO_NO_RECORDS {
		// if there are no aliases, the canonical name is the input name
		return absDomainName([]byte(name)), nil
	}
	if e != nil {
		return "", &DNSError{Err: os.NewSyscallError("dnsquery", e).Error(), Name: name}
	}
	defer syscall.DnsRecordListFree(r, 1)

	resolved := resolveCNAME(syscall.StringToUTF16Ptr(name), r)
	cname := syscall.UTF16ToString((*[256]uint16)(unsafe.Pointer(resolved))[:])
	return absDomainName([]byte(cname)), nil
}

func lookupSRV(service, proto, name string) (string, []*SRV, error) {
	acquireThread()
	defer releaseThread()
	var target string
	if service == "" && proto == "" {
		target = name
	} else {
		target = "_" + service + "._" + proto + "." + name
	}
	var r *syscall.DNSRecord
	e := syscall.DnsQuery(target, syscall.DNS_TYPE_SRV, 0, nil, &r, nil)
	if e != nil {
		return "", nil, &DNSError{Err: os.NewSyscallError("dnsquery", e).Error(), Name: target}
	}
	defer syscall.DnsRecordListFree(r, 1)

	srvs := make([]*SRV, 0, 10)
	for _, p := range validRecs(r, syscall.DNS_TYPE_SRV, target) {
		v := (*syscall.DNSSRVData)(unsafe.Pointer(&p.Data[0]))
		srvs = append(srvs, &SRV{absDomainName([]byte(syscall.UTF16ToString((*[256]uint16)(unsafe.Pointer(v.Target))[:]))), v.Port, v.Priority, v.Weight})
	}
	byPriorityWeight(srvs).sort()
	return absDomainName([]byte(target)), srvs, nil
}

func lookupMX(name string) ([]*MX, error) {
	acquireThread()
	defer releaseThread()
	var r *syscall.DNSRecord
	e := syscall.DnsQuery(name, syscall.DNS_TYPE_MX, 0, nil, &r, nil)
	if e != nil {
		return nil, &DNSError{Err: os.NewSyscallError("dnsquery", e).Error(), Name: name}
	}
	defer syscall.DnsRecordListFree(r, 1)

	mxs := make([]*MX, 0, 10)
	for _, p := range validRecs(r, syscall.DNS_TYPE_MX, name) {
		v := (*syscall.DNSMXData)(unsafe.Pointer(&p.Data[0]))
		mxs = append(mxs, &MX{absDomainName([]byte(syscall.UTF16ToString((*[256]uint16)(unsafe.Pointer(v.NameExchange))[:]))), v.Preference})
	}
	byPref(mxs).sort()
	return mxs, nil
}

func lookupNS(name string) ([]*NS, error) {
	acquireThread()
	defer releaseThread()
	var r *syscall.DNSRecord
	e := syscall.DnsQuery(name, syscall.DNS_TYPE_NS, 0, nil, &r, nil)
	if e != nil {
		return nil, &DNSError{Err: os.NewSyscallError("dnsquery", e).Error(), Name: name}
	}
	defer syscall.DnsRecordListFree(r, 1)

	nss := make([]*NS, 0, 10)
	for _, p := range validRecs(r, syscall.DNS_TYPE_NS, name) {
		v := (*syscall.DNSPTRData)(unsafe.Pointer(&p.Data[0]))
		nss = append(nss, &NS{absDomainName([]byte(syscall.UTF16ToString((*[256]uint16)(unsafe.Pointer(v.Host))[:])))})
	}
	return nss, nil
}

func lookupTXT(name string) ([]string, error) {
	acquireThread()
	defer releaseThread()
	var r *syscall.DNSRecord
	e := syscall.DnsQuery(name, syscall.DNS_TYPE_TEXT, 0, nil, &r, nil)
	if e != nil {
		return nil, &DNSError{Err: os.NewSyscallError("dnsquery", e).Error(), Name: name}
	}
	defer syscall.DnsRecordListFree(r, 1)

	txts := make([]string, 0, 10)
	for _, p := range validRecs(r, syscall.DNS_TYPE_TEXT, name) {
		d := (*syscall.DNSTXTData)(unsafe.Pointer(&p.Data[0]))
		for _, v := range (*[1 << 10]*uint16)(unsafe.Pointer(&(d.StringArray[0])))[:d.StringCount] {
			s := syscall.UTF16ToString((*[1 << 20]uint16)(unsafe.Pointer(v))[:])
			txts = append(txts, s)
		}
	}
	return txts, nil
}

func lookupAddr(addr string) ([]string, error) {
	acquireThread()
	defer releaseThread()
	arpa, err := reverseaddr(addr)
	if err != nil {
		return nil, err
	}
	var r *syscall.DNSRecord
	e := syscall.DnsQuery(arpa, syscall.DNS_TYPE_PTR, 0, nil, &r, nil)
	if e != nil {
		return nil, &DNSError{Err: os.NewSyscallError("dnsquery", e).Error(), Name: addr}
	}
	defer syscall.DnsRecordListFree(r, 1)

	ptrs := make([]string, 0, 10)
	for _, p := range validRecs(r, syscall.DNS_TYPE_PTR, arpa) {
		v := (*syscall.DNSPTRData)(unsafe.Pointer(&p.Data[0]))
		ptrs = append(ptrs, absDomainName([]byte(syscall.UTF16ToString((*[256]uint16)(unsafe.Pointer(v.Host))[:]))))
	}
	return ptrs, nil
}

const dnsSectionMask = 0x0003

// returns only results applicable to name and resolves CNAME entries
func validRecs(r *syscall.DNSRecord, dnstype uint16, name string) []*syscall.DNSRecord {
	cname := syscall.StringToUTF16Ptr(name)
	if dnstype != syscall.DNS_TYPE_CNAME {
		cname = resolveCNAME(cname, r)
	}
	rec := make([]*syscall.DNSRecord, 0, 10)
	for p := r; p != nil; p = p.Next {
		if p.Dw&dnsSectionMask != syscall.DnsSectionAnswer {
			continue
		}
		if p.Type != dnstype {
			continue
		}
		if !syscall.DnsNameCompare(cname, p.Name) {
			continue
		}
		rec = append(rec, p)
	}
	return rec
}

// returns the last CNAME in chain
func resolveCNAME(name *uint16, r *syscall.DNSRecord) *uint16 {
	// limit cname resolving to 10 in case of a infinite CNAME loop
Cname:
	for cnameloop := 0; cnameloop < 10; cnameloop++ {
		for p := r; p != nil; p = p.Next {
			if p.Dw&dnsSectionMask != syscall.DnsSectionAnswer {
				continue
			}
			if p.Type != syscall.DNS_TYPE_CNAME {
				continue
			}
			if !syscall.DnsNameCompare(name, p.Name) {
				continue
			}
			name = (*syscall.DNSPTRData)(unsafe.Pointer(&r.Data[0])).Host
			continue Cname
		}
		break
	}
	return name
}

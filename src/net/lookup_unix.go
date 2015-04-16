// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package net

import "sync"

var onceReadProtocols sync.Once

// readProtocols loads contents of /etc/protocols into protocols map
// for quick access.
func readProtocols() {
	if file, err := open("/etc/protocols"); err == nil {
		for line, ok := file.readLine(); ok; line, ok = file.readLine() {
			// tcp    6   TCP    # transmission control protocol
			if i := byteIndex(line, '#'); i >= 0 {
				line = line[0:i]
			}
			f := getFields(line)
			if len(f) < 2 {
				continue
			}
			if proto, _, ok := dtoi(f[1], 0); ok {
				if _, ok := protocols[f[0]]; !ok {
					protocols[f[0]] = proto
				}
				for _, alias := range f[2:] {
					if _, ok := protocols[alias]; !ok {
						protocols[alias] = proto
					}
				}
			}
		}
		file.close()
	}
}

// lookupProtocol looks up IP protocol name in /etc/protocols and
// returns correspondent protocol number.
func lookupProtocol(name string) (int, error) {
	onceReadProtocols.Do(readProtocols)
	proto, found := protocols[name]
	if !found {
		return 0, &AddrError{Err: "unknown IP protocol specified", Addr: name}
	}
	return proto, nil
}

func lookupHost(host string) (addrs []string, err error) {
	order := systemConf().hostLookupOrder(host)
	if order == hostLookupCgo {
		if addrs, err, ok := cgoLookupHost(host); ok {
			return addrs, err
		}
		// cgo not available (or netgo); fall back to Go's DNS resolver
		order = hostLookupFilesDNS
	}
	return goLookupHostOrder(host, order)
}

func lookupIP(host string) (addrs []IPAddr, err error) {
	order := systemConf().hostLookupOrder(host)
	if order == hostLookupCgo {
		if addrs, err, ok := cgoLookupIP(host); ok {
			return addrs, err
		}
		// cgo not available (or netgo); fall back to Go's DNS resolver
		order = hostLookupFilesDNS
	}
	return goLookupIPOrder(host, order)
}

func lookupPort(network, service string) (int, error) {
	port, err, ok := cgoLookupPort(network, service)
	if !ok {
		port, err = goLookupPort(network, service)
	}
	return port, err
}

func lookupCNAME(name string) (string, error) {
	cname, err, ok := cgoLookupCNAME(name)
	if !ok {
		cname, err = goLookupCNAME(name)
	}
	return cname, err
}

func lookupSRV(service, proto, name string) (string, []*SRV, error) {
	var target string
	if service == "" && proto == "" {
		target = name
	} else {
		target = "_" + service + "._" + proto + "." + name
	}
	cname, rrs, err := lookup(target, dnsTypeSRV)
	if err != nil {
		return "", nil, err
	}
	srvs := make([]*SRV, len(rrs))
	for i, rr := range rrs {
		rr := rr.(*dnsRR_SRV)
		srvs[i] = &SRV{Target: rr.Target, Port: rr.Port, Priority: rr.Priority, Weight: rr.Weight}
	}
	byPriorityWeight(srvs).sort()
	return cname, srvs, nil
}

func lookupMX(name string) ([]*MX, error) {
	_, rrs, err := lookup(name, dnsTypeMX)
	if err != nil {
		return nil, err
	}
	mxs := make([]*MX, len(rrs))
	for i, rr := range rrs {
		rr := rr.(*dnsRR_MX)
		mxs[i] = &MX{Host: rr.Mx, Pref: rr.Pref}
	}
	byPref(mxs).sort()
	return mxs, nil
}

func lookupNS(name string) ([]*NS, error) {
	_, rrs, err := lookup(name, dnsTypeNS)
	if err != nil {
		return nil, err
	}
	nss := make([]*NS, len(rrs))
	for i, rr := range rrs {
		nss[i] = &NS{Host: rr.(*dnsRR_NS).Ns}
	}
	return nss, nil
}

func lookupTXT(name string) ([]string, error) {
	_, rrs, err := lookup(name, dnsTypeTXT)
	if err != nil {
		return nil, err
	}
	txts := make([]string, len(rrs))
	for i, rr := range rrs {
		txts[i] = rr.(*dnsRR_TXT).Txt
	}
	return txts, nil
}

func lookupAddr(addr string) ([]string, error) {
	names := lookupStaticAddr(addr)
	if len(names) > 0 {
		return names, nil
	}
	arpa, err := reverseaddr(addr)
	if err != nil {
		return nil, err
	}
	_, rrs, err := lookup(arpa, dnsTypePTR)
	if err != nil {
		return nil, err
	}
	ptrs := make([]string, len(rrs))
	for i, rr := range rrs {
		ptrs[i] = rr.(*dnsRR_PTR).Ptr
	}
	return ptrs, nil
}

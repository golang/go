// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd

package net

import (
	"errors"
	"sync"
)

var (
	protocols         map[string]int
	onceReadProtocols sync.Once
)

// readProtocols loads contents of /etc/protocols into protocols map
// for quick access.
func readProtocols() {
	protocols = make(map[string]int)
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
				protocols[f[0]] = proto
				for _, alias := range f[2:] {
					protocols[alias] = proto
				}
			}
		}
		file.close()
	}
}

// lookupProtocol looks up IP protocol name in /etc/protocols and
// returns correspondent protocol number.
func lookupProtocol(name string) (proto int, err error) {
	onceReadProtocols.Do(readProtocols)
	proto, found := protocols[name]
	if !found {
		return 0, errors.New("unknown IP protocol specified: " + name)
	}
	return
}

func lookupHost(host string) (addrs []string, err error) {
	addrs, err, ok := cgoLookupHost(host)
	if !ok {
		addrs, err = goLookupHost(host)
	}
	return
}

func lookupIP(host string) (addrs []IP, err error) {
	addrs, err, ok := cgoLookupIP(host)
	if !ok {
		addrs, err = goLookupIP(host)
	}
	return
}

func lookupPort(network, service string) (port int, err error) {
	port, err, ok := cgoLookupPort(network, service)
	if !ok {
		port, err = goLookupPort(network, service)
	}
	return
}

func lookupCNAME(name string) (cname string, err error) {
	cname, err, ok := cgoLookupCNAME(name)
	if !ok {
		cname, err = goLookupCNAME(name)
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
	var records []dnsRR
	cname, records, err = lookup(target, dnsTypeSRV)
	if err != nil {
		return
	}
	addrs = make([]*SRV, len(records))
	for i, rr := range records {
		r := rr.(*dnsRR_SRV)
		addrs[i] = &SRV{r.Target, r.Port, r.Priority, r.Weight}
	}
	byPriorityWeight(addrs).sort()
	return
}

func lookupMX(name string) (mx []*MX, err error) {
	_, records, err := lookup(name, dnsTypeMX)
	if err != nil {
		return
	}
	mx = make([]*MX, len(records))
	for i, rr := range records {
		r := rr.(*dnsRR_MX)
		mx[i] = &MX{r.Mx, r.Pref}
	}
	byPref(mx).sort()
	return
}

func lookupNS(name string) (ns []*NS, err error) {
	_, records, err := lookup(name, dnsTypeNS)
	if err != nil {
		return
	}
	ns = make([]*NS, len(records))
	for i, r := range records {
		r := r.(*dnsRR_NS)
		ns[i] = &NS{r.Ns}
	}
	return
}

func lookupTXT(name string) (txt []string, err error) {
	_, records, err := lookup(name, dnsTypeTXT)
	if err != nil {
		return
	}
	txt = make([]string, len(records))
	for i, r := range records {
		txt[i] = r.(*dnsRR_TXT).Txt
	}
	return
}

func lookupAddr(addr string) (name []string, err error) {
	name = lookupStaticAddr(addr)
	if len(name) > 0 {
		return
	}
	var arpa string
	arpa, err = reverseaddr(addr)
	if err != nil {
		return
	}
	var records []dnsRR
	_, records, err = lookup(arpa, dnsTypePTR)
	if err != nil {
		return
	}
	name = make([]string, len(records))
	for i := range records {
		r := records[i].(*dnsRR_PTR)
		name[i] = r.Ptr
	}
	return
}

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"os"
)

// LookupHost looks up the given host using the local resolver.
// It returns an array of that host's addresses.
func LookupHost(host string) (addrs []string, err os.Error) {
	addrs, err, ok := cgoLookupHost(host)
	if !ok {
		addrs, err = goLookupHost(host)
	}
	return
}

// LookupIP looks up host using the local resolver.
// It returns an array of that host's IPv4 and IPv6 addresses.
func LookupIP(host string) (addrs []IP, err os.Error) {
	addrs, err, ok := cgoLookupIP(host)
	if !ok {
		addrs, err = goLookupIP(host)
	}
	return
}

// LookupPort looks up the port for the given network and service.
func LookupPort(network, service string) (port int, err os.Error) {
	port, err, ok := cgoLookupPort(network, service)
	if !ok {
		port, err = goLookupPort(network, service)
	}
	return
}

// LookupCNAME returns the canonical DNS host for the given name.
// Callers that do not care about the canonical name can call
// LookupHost or LookupIP directly; both take care of resolving
// the canonical name as part of the lookup.
func LookupCNAME(name string) (cname string, err os.Error) {
	cname, err, ok := cgoLookupCNAME(name)
	if !ok {
		cname, err = goLookupCNAME(name)
	}
	return
}

// LookupSRV tries to resolve an SRV query of the given service,
// protocol, and domain name, as specified in RFC 2782. In most cases
// the proto argument can be the same as the corresponding
// Addr.Network(). The returned records are sorted by priority 
// and randomized by weight within a priority.
func LookupSRV(service, proto, name string) (cname string, addrs []*SRV, err os.Error) {
	target := "_" + service + "._" + proto + "." + name
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

// LookupMX returns the DNS MX records for the given domain name sorted by preference.
func LookupMX(name string) (mx []*MX, err os.Error) {
	_, rr, err := lookup(name, dnsTypeMX)
	if err != nil {
		return
	}
	mx = make([]*MX, len(rr))
	for i := range rr {
		r := rr[i].(*dnsRR_MX)
		mx[i] = &MX{r.Mx, r.Pref}
	}
	byPref(mx).sort()
	return
}

// LookupAddr performs a reverse lookup for the given address, returning a list
// of names mapping to that address.
func LookupAddr(addr string) (name []string, err os.Error) {
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

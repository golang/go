// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"context"
	"internal/nettrace"
	"internal/singleflight"
)

// protocols contains minimal mappings between internet protocol
// names and numbers for platforms that don't have a complete list of
// protocol numbers.
//
// See http://www.iana.org/assignments/protocol-numbers
var protocols = map[string]int{
	"icmp": 1, "ICMP": 1,
	"igmp": 2, "IGMP": 2,
	"tcp": 6, "TCP": 6,
	"udp": 17, "UDP": 17,
	"ipv6-icmp": 58, "IPV6-ICMP": 58, "IPv6-ICMP": 58,
}

// LookupHost looks up the given host using the local resolver.
// It returns an array of that host's addresses.
func LookupHost(host string) (addrs []string, err error) {
	// Make sure that no matter what we do later, host=="" is rejected.
	// ParseIP, for example, does accept empty strings.
	if host == "" {
		return nil, &DNSError{Err: errNoSuchHost.Error(), Name: host}
	}
	if ip := ParseIP(host); ip != nil {
		return []string{host}, nil
	}
	return lookupHost(context.Background(), host)
}

// LookupIP looks up host using the local resolver.
// It returns an array of that host's IPv4 and IPv6 addresses.
func LookupIP(host string) (ips []IP, err error) {
	// Make sure that no matter what we do later, host=="" is rejected.
	// ParseIP, for example, does accept empty strings.
	if host == "" {
		return nil, &DNSError{Err: errNoSuchHost.Error(), Name: host}
	}
	if ip := ParseIP(host); ip != nil {
		return []IP{ip}, nil
	}
	addrs, err := lookupIPMerge(context.Background(), host)
	if err != nil {
		return
	}
	ips = make([]IP, len(addrs))
	for i, addr := range addrs {
		ips[i] = addr.IP
	}
	return
}

var lookupGroup singleflight.Group

// lookupIPMerge wraps lookupIP, but makes sure that for any given
// host, only one lookup is in-flight at a time. The returned memory
// is always owned by the caller.
func lookupIPMerge(ctx context.Context, host string) (addrs []IPAddr, err error) {
	addrsi, err, shared := lookupGroup.Do(host, func() (interface{}, error) {
		return testHookLookupIP(ctx, lookupIP, host)
	})
	return lookupIPReturn(addrsi, err, shared)
}

// lookupIPReturn turns the return values from singleflight.Do into
// the return values from LookupIP.
func lookupIPReturn(addrsi interface{}, err error, shared bool) ([]IPAddr, error) {
	if err != nil {
		return nil, err
	}
	addrs := addrsi.([]IPAddr)
	if shared {
		clone := make([]IPAddr, len(addrs))
		copy(clone, addrs)
		addrs = clone
	}
	return addrs, nil
}

// ipAddrsEface returns an empty interface slice of addrs.
func ipAddrsEface(addrs []IPAddr) []interface{} {
	s := make([]interface{}, len(addrs))
	for i, v := range addrs {
		s[i] = v
	}
	return s
}

// lookupIPContext looks up a hostname with a context.
//
// TODO(bradfitz): rename this function. All the other
// build-tag-specific lookupIP funcs also take a context now, so this
// name is no longer great. Maybe make this lookupIPMerge and ditch
// the other one, making its callers call this instead with a
// context.Background().
func lookupIPContext(ctx context.Context, host string) (addrs []IPAddr, err error) {
	trace, _ := ctx.Value(nettrace.TraceKey{}).(*nettrace.Trace)
	if trace != nil && trace.DNSStart != nil {
		trace.DNSStart(host)
	}
	// The underlying resolver func is lookupIP by default but it
	// can be overridden by tests. This is needed by net/http, so it
	// uses a context key instead of unexported variables.
	resolverFunc := lookupIP
	if alt, _ := ctx.Value(nettrace.LookupIPAltResolverKey{}).(func(context.Context, string) ([]IPAddr, error)); alt != nil {
		resolverFunc = alt
	}

	ch := lookupGroup.DoChan(host, func() (interface{}, error) {
		return testHookLookupIP(ctx, resolverFunc, host)
	})

	select {
	case <-ctx.Done():
		// The DNS lookup timed out for some reason. Force
		// future requests to start the DNS lookup again
		// rather than waiting for the current lookup to
		// complete. See issue 8602.
		err := mapErr(ctx.Err())
		lookupGroup.Forget(host)
		if trace != nil && trace.DNSDone != nil {
			trace.DNSDone(nil, false, err)
		}
		return nil, err
	case r := <-ch:
		if trace != nil && trace.DNSDone != nil {
			addrs, _ := r.Val.([]IPAddr)
			trace.DNSDone(ipAddrsEface(addrs), r.Shared, r.Err)
		}
		return lookupIPReturn(r.Val, r.Err, r.Shared)
	}
}

// LookupPort looks up the port for the given network and service.
func LookupPort(network, service string) (port int, err error) {
	port, needsLookup := parsePort(service)
	if needsLookup {
		port, err = lookupPort(context.Background(), network, service)
		if err != nil {
			return 0, err
		}
	}
	if 0 > port || port > 65535 {
		return 0, &AddrError{Err: "invalid port", Addr: service}
	}
	return port, nil
}

// LookupCNAME returns the canonical DNS host for the given name.
// Callers that do not care about the canonical name can call
// LookupHost or LookupIP directly; both take care of resolving
// the canonical name as part of the lookup.
func LookupCNAME(name string) (cname string, err error) {
	return lookupCNAME(context.Background(), name)
}

// LookupSRV tries to resolve an SRV query of the given service,
// protocol, and domain name. The proto is "tcp" or "udp".
// The returned records are sorted by priority and randomized
// by weight within a priority.
//
// LookupSRV constructs the DNS name to look up following RFC 2782.
// That is, it looks up _service._proto.name. To accommodate services
// publishing SRV records under non-standard names, if both service
// and proto are empty strings, LookupSRV looks up name directly.
func LookupSRV(service, proto, name string) (cname string, addrs []*SRV, err error) {
	return lookupSRV(context.Background(), service, proto, name)
}

// LookupMX returns the DNS MX records for the given domain name sorted by preference.
func LookupMX(name string) (mxs []*MX, err error) {
	return lookupMX(context.Background(), name)
}

// LookupNS returns the DNS NS records for the given domain name.
func LookupNS(name string) (nss []*NS, err error) {
	return lookupNS(context.Background(), name)
}

// LookupTXT returns the DNS TXT records for the given domain name.
func LookupTXT(name string) (txts []string, err error) {
	return lookupTXT(context.Background(), name)
}

// LookupAddr performs a reverse lookup for the given address, returning a list
// of names mapping to that address.
func LookupAddr(addr string) (names []string, err error) {
	return lookupAddr(context.Background(), addr)
}

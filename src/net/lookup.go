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
//
// On Unix, this map is augmented by readProtocols via lookupProtocol.
var protocols = map[string]int{
	"icmp":      1,
	"igmp":      2,
	"tcp":       6,
	"udp":       17,
	"ipv6-icmp": 58,
}

// services contains minimal mappings between services names and port
// numbers for platforms that don't have a complete list of port numbers
// (some Solaris distros, nacl, etc).
//
// See https://www.iana.org/assignments/service-names-port-numbers
//
// On Unix, this map is augmented by readServices via goLookupPort.
var services = map[string]map[string]int{
	"udp": {
		"domain": 53,
	},
	"tcp": {
		"ftp":    21,
		"ftps":   990,
		"gopher": 70, // ʕ◔ϖ◔ʔ
		"http":   80,
		"https":  443,
		"imap2":  143,
		"imap3":  220,
		"imaps":  993,
		"pop3":   110,
		"pop3s":  995,
		"smtp":   25,
		"ssh":    22,
		"telnet": 23,
	},
}

const maxProtoLength = len("RSVP-E2E-IGNORE") + 10 // with room to grow

func lookupProtocolMap(name string) (int, error) {
	var lowerProtocol [maxProtoLength]byte
	n := copy(lowerProtocol[:], name)
	lowerASCIIBytes(lowerProtocol[:n])
	proto, found := protocols[string(lowerProtocol[:n])]
	if !found || n != len(name) {
		return 0, &AddrError{Err: "unknown IP protocol specified", Addr: name}
	}
	return proto, nil
}

// maxPortBufSize is the longest reasonable name of a service
// (non-numeric port).
// Currently the longest known IANA-unregistered name is
// "mobility-header", so we use that length, plus some slop in case
// something longer is added in the future.
const maxPortBufSize = len("mobility-header") + 10

func lookupPortMap(network, service string) (port int, error error) {
	switch network {
	case "tcp4", "tcp6":
		network = "tcp"
	case "udp4", "udp6":
		network = "udp"
	}

	if m, ok := services[network]; ok {
		var lowerService [maxPortBufSize]byte
		n := copy(lowerService[:], service)
		lowerASCIIBytes(lowerService[:n])
		if port, ok := m[string(lowerService[:n])]; ok && n == len(service) {
			return port, nil
		}
	}
	return 0, &AddrError{Err: "unknown port", Addr: network + "/" + service}
}

// DefaultResolver is the resolver used by the package-level Lookup
// functions and by Dialers without a specified Resolver.
var DefaultResolver = &Resolver{}

// A Resolver looks up names and numbers.
//
// A nil *Resolver is equivalent to a zero Resolver.
type Resolver struct {
	// PreferGo controls whether Go's built-in DNS resolver is preferred
	// on platforms where it's available. It is equivalent to setting
	// GODEBUG=netdns=go, but scoped to just this resolver.
	PreferGo bool

	// StrictErrors controls the behavior of temporary errors
	// (including timeout, socket errors, and SERVFAIL) when using
	// Go's built-in resolver. For a query composed of multiple
	// sub-queries (such as an A+AAAA address lookup, or walking the
	// DNS search list), this option causes such errors to abort the
	// whole query instead of returning a partial result. This is
	// not enabled by default because it may affect compatibility
	// with resolvers that process AAAA queries incorrectly.
	StrictErrors bool

	// Dial optionally specifies an alternate dialer for use by
	// Go's built-in DNS resolver to make TCP and UDP connections
	// to DNS services. The host in the address parameter will
	// always be a literal IP address and not a host name, and the
	// port in the address parameter will be a literal port number
	// and not a service name.
	// If the Conn returned is also a PacketConn, sent and received DNS
	// messages must adhere to RFC 1035 section 4.2.1, "UDP usage".
	// Otherwise, DNS messages transmitted over Conn must adhere
	// to RFC 7766 section 5, "Transport Protocol Selection".
	// If nil, the default dialer is used.
	Dial func(ctx context.Context, network, address string) (Conn, error)

	// TODO(bradfitz): optional interface impl override hook
	// TODO(bradfitz): Timeout time.Duration?
}

// LookupHost looks up the given host using the local resolver.
// It returns a slice of that host's addresses.
func LookupHost(host string) (addrs []string, err error) {
	return DefaultResolver.LookupHost(context.Background(), host)
}

// LookupHost looks up the given host using the local resolver.
// It returns a slice of that host's addresses.
func (r *Resolver) LookupHost(ctx context.Context, host string) (addrs []string, err error) {
	// Make sure that no matter what we do later, host=="" is rejected.
	// ParseIP, for example, does accept empty strings.
	if host == "" {
		return nil, &DNSError{Err: errNoSuchHost.Error(), Name: host}
	}
	if ip := ParseIP(host); ip != nil {
		return []string{host}, nil
	}
	return r.lookupHost(ctx, host)
}

// LookupIP looks up host using the local resolver.
// It returns a slice of that host's IPv4 and IPv6 addresses.
func LookupIP(host string) ([]IP, error) {
	addrs, err := DefaultResolver.LookupIPAddr(context.Background(), host)
	if err != nil {
		return nil, err
	}
	ips := make([]IP, len(addrs))
	for i, ia := range addrs {
		ips[i] = ia.IP
	}
	return ips, nil
}

// LookupIPAddr looks up host using the local resolver.
// It returns a slice of that host's IPv4 and IPv6 addresses.
func (r *Resolver) LookupIPAddr(ctx context.Context, host string) ([]IPAddr, error) {
	// Make sure that no matter what we do later, host=="" is rejected.
	// ParseIP, for example, does accept empty strings.
	if host == "" {
		return nil, &DNSError{Err: errNoSuchHost.Error(), Name: host}
	}
	if ip := ParseIP(host); ip != nil {
		return []IPAddr{{IP: ip}}, nil
	}
	trace, _ := ctx.Value(nettrace.TraceKey{}).(*nettrace.Trace)
	if trace != nil && trace.DNSStart != nil {
		trace.DNSStart(host)
	}
	// The underlying resolver func is lookupIP by default but it
	// can be overridden by tests. This is needed by net/http, so it
	// uses a context key instead of unexported variables.
	resolverFunc := r.lookupIP
	if alt, _ := ctx.Value(nettrace.LookupIPAltResolverKey{}).(func(context.Context, string) ([]IPAddr, error)); alt != nil {
		resolverFunc = alt
	}

	ch := lookupGroup.DoChan(host, func() (interface{}, error) {
		return testHookLookupIP(ctx, resolverFunc, host)
	})

	select {
	case <-ctx.Done():
		// If the DNS lookup timed out for some reason, force
		// future requests to start the DNS lookup again
		// rather than waiting for the current lookup to
		// complete. See issue 8602.
		ctxErr := ctx.Err()
		if ctxErr == context.DeadlineExceeded {
			lookupGroup.Forget(host)
		}
		err := mapErr(ctxErr)
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

// lookupGroup merges LookupIPAddr calls together for lookups
// for the same host. The lookupGroup key is is the LookupIPAddr.host
// argument.
// The return values are ([]IPAddr, error).
var lookupGroup singleflight.Group

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

// LookupPort looks up the port for the given network and service.
func LookupPort(network, service string) (port int, err error) {
	return DefaultResolver.LookupPort(context.Background(), network, service)
}

// LookupPort looks up the port for the given network and service.
func (r *Resolver) LookupPort(ctx context.Context, network, service string) (port int, err error) {
	port, needsLookup := parsePort(service)
	if needsLookup {
		port, err = r.lookupPort(ctx, network, service)
		if err != nil {
			return 0, err
		}
	}
	if 0 > port || port > 65535 {
		return 0, &AddrError{Err: "invalid port", Addr: service}
	}
	return port, nil
}

// LookupCNAME returns the canonical name for the given host.
// Callers that do not care about the canonical name can call
// LookupHost or LookupIP directly; both take care of resolving
// the canonical name as part of the lookup.
//
// A canonical name is the final name after following zero
// or more CNAME records.
// LookupCNAME does not return an error if host does not
// contain DNS "CNAME" records, as long as host resolves to
// address records.
func LookupCNAME(host string) (cname string, err error) {
	return DefaultResolver.lookupCNAME(context.Background(), host)
}

// LookupCNAME returns the canonical name for the given host.
// Callers that do not care about the canonical name can call
// LookupHost or LookupIP directly; both take care of resolving
// the canonical name as part of the lookup.
//
// A canonical name is the final name after following zero
// or more CNAME records.
// LookupCNAME does not return an error if host does not
// contain DNS "CNAME" records, as long as host resolves to
// address records.
func (r *Resolver) LookupCNAME(ctx context.Context, host string) (cname string, err error) {
	return r.lookupCNAME(ctx, host)
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
	return DefaultResolver.lookupSRV(context.Background(), service, proto, name)
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
func (r *Resolver) LookupSRV(ctx context.Context, service, proto, name string) (cname string, addrs []*SRV, err error) {
	return r.lookupSRV(ctx, service, proto, name)
}

// LookupMX returns the DNS MX records for the given domain name sorted by preference.
func LookupMX(name string) ([]*MX, error) {
	return DefaultResolver.lookupMX(context.Background(), name)
}

// LookupMX returns the DNS MX records for the given domain name sorted by preference.
func (r *Resolver) LookupMX(ctx context.Context, name string) ([]*MX, error) {
	return r.lookupMX(ctx, name)
}

// LookupNS returns the DNS NS records for the given domain name.
func LookupNS(name string) ([]*NS, error) {
	return DefaultResolver.lookupNS(context.Background(), name)
}

// LookupNS returns the DNS NS records for the given domain name.
func (r *Resolver) LookupNS(ctx context.Context, name string) ([]*NS, error) {
	return r.lookupNS(ctx, name)
}

// LookupTXT returns the DNS TXT records for the given domain name.
func LookupTXT(name string) ([]string, error) {
	return DefaultResolver.lookupTXT(context.Background(), name)
}

// LookupTXT returns the DNS TXT records for the given domain name.
func (r *Resolver) LookupTXT(ctx context.Context, name string) ([]string, error) {
	return r.lookupTXT(ctx, name)
}

// LookupAddr performs a reverse lookup for the given address, returning a list
// of names mapping to that address.
//
// When using the host C library resolver, at most one result will be
// returned. To bypass the host resolver, use a custom Resolver.
func LookupAddr(addr string) (names []string, err error) {
	return DefaultResolver.lookupAddr(context.Background(), addr)
}

// LookupAddr performs a reverse lookup for the given address, returning a list
// of names mapping to that address.
func (r *Resolver) LookupAddr(ctx context.Context, addr string) (names []string, err error) {
	return r.lookupAddr(ctx, addr)
}

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

// DNS client: see RFC 1035.
// Has to be linked into package net for Dial.

// TODO(rsc):
//	Could potentially handle many outstanding lookups faster.
//	Could have a small cache.
//	Random UDP source port (net.Dial should do that for us).
//	Random request IDs.

package net

import (
	"context"
	"errors"
	"io"
	"math/rand"
	"os"
	"sync"
	"time"
)

// A dnsDialer provides dialing suitable for DNS queries.
type dnsDialer interface {
	dialDNS(ctx context.Context, network, addr string) (dnsConn, error)
}

var testHookDNSDialer = func() dnsDialer { return &Dialer{} }

// A dnsConn represents a DNS transport endpoint.
type dnsConn interface {
	io.Closer

	SetDeadline(time.Time) error

	// dnsRoundTrip executes a single DNS transaction, returning a
	// DNS response message for the provided DNS query message.
	dnsRoundTrip(query *dnsMsg) (*dnsMsg, error)
}

func (c *UDPConn) dnsRoundTrip(query *dnsMsg) (*dnsMsg, error) {
	return dnsRoundTripUDP(c, query)
}

// dnsRoundTripUDP implements the dnsRoundTrip interface for RFC 1035's
// "UDP usage" transport mechanism. c should be a packet-oriented connection,
// such as a *UDPConn.
func dnsRoundTripUDP(c io.ReadWriter, query *dnsMsg) (*dnsMsg, error) {
	b, ok := query.Pack()
	if !ok {
		return nil, errors.New("cannot marshal DNS message")
	}
	if _, err := c.Write(b); err != nil {
		return nil, err
	}

	b = make([]byte, 512) // see RFC 1035
	for {
		n, err := c.Read(b)
		if err != nil {
			return nil, err
		}
		resp := &dnsMsg{}
		if !resp.Unpack(b[:n]) || !resp.IsResponseTo(query) {
			// Ignore invalid responses as they may be malicious
			// forgery attempts. Instead continue waiting until
			// timeout. See golang.org/issue/13281.
			continue
		}
		return resp, nil
	}
}

func (c *TCPConn) dnsRoundTrip(out *dnsMsg) (*dnsMsg, error) {
	return dnsRoundTripTCP(c, out)
}

// dnsRoundTripTCP implements the dnsRoundTrip interface for RFC 1035's
// "TCP usage" transport mechanism. c should be a stream-oriented connection,
// such as a *TCPConn.
func dnsRoundTripTCP(c io.ReadWriter, query *dnsMsg) (*dnsMsg, error) {
	b, ok := query.Pack()
	if !ok {
		return nil, errors.New("cannot marshal DNS message")
	}
	l := len(b)
	b = append([]byte{byte(l >> 8), byte(l)}, b...)
	if _, err := c.Write(b); err != nil {
		return nil, err
	}

	b = make([]byte, 1280) // 1280 is a reasonable initial size for IP over Ethernet, see RFC 4035
	if _, err := io.ReadFull(c, b[:2]); err != nil {
		return nil, err
	}
	l = int(b[0])<<8 | int(b[1])
	if l > len(b) {
		b = make([]byte, l)
	}
	n, err := io.ReadFull(c, b[:l])
	if err != nil {
		return nil, err
	}
	resp := &dnsMsg{}
	if !resp.Unpack(b[:n]) {
		return nil, errors.New("cannot unmarshal DNS message")
	}
	if !resp.IsResponseTo(query) {
		return nil, errors.New("invalid DNS response")
	}
	return resp, nil
}

func (d *Dialer) dialDNS(ctx context.Context, network, server string) (dnsConn, error) {
	switch network {
	case "tcp", "tcp4", "tcp6", "udp", "udp4", "udp6":
	default:
		return nil, UnknownNetworkError(network)
	}
	// Calling Dial here is scary -- we have to be sure not to
	// dial a name that will require a DNS lookup, or Dial will
	// call back here to translate it. The DNS config parser has
	// already checked that all the cfg.servers[i] are IP
	// addresses, which Dial will use without a DNS lookup.
	c, err := d.DialContext(ctx, network, server)
	if err != nil {
		return nil, mapErr(err)
	}
	switch network {
	case "tcp", "tcp4", "tcp6":
		return c.(*TCPConn), nil
	case "udp", "udp4", "udp6":
		return c.(*UDPConn), nil
	}
	panic("unreachable")
}

// exchange sends a query on the connection and hopes for a response.
func exchange(ctx context.Context, server, name string, qtype uint16, timeout time.Duration) (*dnsMsg, error) {
	d := testHookDNSDialer()
	out := dnsMsg{
		dnsMsgHdr: dnsMsgHdr{
			recursion_desired: true,
		},
		question: []dnsQuestion{
			{name, qtype, dnsClassINET},
		},
	}
	for _, network := range []string{"udp", "tcp"} {
		// TODO(mdempsky): Refactor so defers from UDP-based
		// exchanges happen before TCP-based exchange.

		ctx, cancel := context.WithDeadline(ctx, time.Now().Add(timeout))
		defer cancel()

		c, err := d.dialDNS(ctx, network, server)
		if err != nil {
			return nil, err
		}
		defer c.Close()
		if d, ok := ctx.Deadline(); ok && !d.IsZero() {
			c.SetDeadline(d)
		}
		out.id = uint16(rand.Int()) ^ uint16(time.Now().UnixNano())
		in, err := c.dnsRoundTrip(&out)
		if err != nil {
			return nil, mapErr(err)
		}
		if in.truncated { // see RFC 5966
			continue
		}
		return in, nil
	}
	return nil, errors.New("no answer from DNS server")
}

// Do a lookup for a single name, which must be rooted
// (otherwise answer will not find the answers).
func tryOneName(ctx context.Context, cfg *dnsConfig, name string, qtype uint16) (string, []dnsRR, error) {
	if len(cfg.servers) == 0 {
		return "", nil, &DNSError{Err: "no DNS servers", Name: name}
	}

	var lastErr error
	for i := 0; i < cfg.attempts; i++ {
		for _, server := range cfg.servers {
			msg, err := exchange(ctx, server, name, qtype, cfg.timeout)
			if err != nil {
				lastErr = &DNSError{
					Err:    err.Error(),
					Name:   name,
					Server: server,
				}
				if nerr, ok := err.(Error); ok && nerr.Timeout() {
					lastErr.(*DNSError).IsTimeout = true
				}
				continue
			}
			// libresolv continues to the next server when it receives
			// an invalid referral response. See golang.org/issue/15434.
			if msg.rcode == dnsRcodeSuccess && !msg.authoritative && !msg.recursion_available && len(msg.answer) == 0 && len(msg.extra) == 0 {
				lastErr = &DNSError{Err: "lame referral", Name: name, Server: server}
				continue
			}
			cname, rrs, err := answer(name, server, msg, qtype)
			// If answer errored for rcodes dnsRcodeSuccess or dnsRcodeNameError,
			// it means the response in msg was not useful and trying another
			// server probably won't help. Return now in those cases.
			// TODO: indicate this in a more obvious way, such as a field on DNSError?
			if err == nil || msg.rcode == dnsRcodeSuccess || msg.rcode == dnsRcodeNameError {
				return cname, rrs, err
			}
			lastErr = err
		}
	}
	return "", nil, lastErr
}

// addrRecordList converts and returns a list of IP addresses from DNS
// address records (both A and AAAA). Other record types are ignored.
func addrRecordList(rrs []dnsRR) []IPAddr {
	addrs := make([]IPAddr, 0, 4)
	for _, rr := range rrs {
		switch rr := rr.(type) {
		case *dnsRR_A:
			addrs = append(addrs, IPAddr{IP: IPv4(byte(rr.A>>24), byte(rr.A>>16), byte(rr.A>>8), byte(rr.A))})
		case *dnsRR_AAAA:
			ip := make(IP, IPv6len)
			copy(ip, rr.AAAA[:])
			addrs = append(addrs, IPAddr{IP: ip})
		}
	}
	return addrs
}

// A resolverConfig represents a DNS stub resolver configuration.
type resolverConfig struct {
	initOnce sync.Once // guards init of resolverConfig

	// ch is used as a semaphore that only allows one lookup at a
	// time to recheck resolv.conf.
	ch          chan struct{} // guards lastChecked and modTime
	lastChecked time.Time     // last time resolv.conf was checked

	mu        sync.RWMutex // protects dnsConfig
	dnsConfig *dnsConfig   // parsed resolv.conf structure used in lookups
}

var resolvConf resolverConfig

// init initializes conf and is only called via conf.initOnce.
func (conf *resolverConfig) init() {
	// Set dnsConfig and lastChecked so we don't parse
	// resolv.conf twice the first time.
	conf.dnsConfig = systemConf().resolv
	if conf.dnsConfig == nil {
		conf.dnsConfig = dnsReadConfig("/etc/resolv.conf")
	}
	conf.lastChecked = time.Now()

	// Prepare ch so that only one update of resolverConfig may
	// run at once.
	conf.ch = make(chan struct{}, 1)
}

// tryUpdate tries to update conf with the named resolv.conf file.
// The name variable only exists for testing. It is otherwise always
// "/etc/resolv.conf".
func (conf *resolverConfig) tryUpdate(name string) {
	conf.initOnce.Do(conf.init)

	// Ensure only one update at a time checks resolv.conf.
	if !conf.tryAcquireSema() {
		return
	}
	defer conf.releaseSema()

	now := time.Now()
	if conf.lastChecked.After(now.Add(-5 * time.Second)) {
		return
	}
	conf.lastChecked = now

	var mtime time.Time
	if fi, err := os.Stat(name); err == nil {
		mtime = fi.ModTime()
	}
	if mtime.Equal(conf.dnsConfig.mtime) {
		return
	}

	dnsConf := dnsReadConfig(name)
	conf.mu.Lock()
	conf.dnsConfig = dnsConf
	conf.mu.Unlock()
}

func (conf *resolverConfig) tryAcquireSema() bool {
	select {
	case conf.ch <- struct{}{}:
		return true
	default:
		return false
	}
}

func (conf *resolverConfig) releaseSema() {
	<-conf.ch
}

func lookup(ctx context.Context, name string, qtype uint16) (cname string, rrs []dnsRR, err error) {
	if !isDomainName(name) {
		return "", nil, &DNSError{Err: "invalid domain name", Name: name}
	}
	resolvConf.tryUpdate("/etc/resolv.conf")
	resolvConf.mu.RLock()
	conf := resolvConf.dnsConfig
	resolvConf.mu.RUnlock()
	for _, fqdn := range conf.nameList(name) {
		cname, rrs, err = tryOneName(ctx, conf, fqdn, qtype)
		if err == nil {
			break
		}
	}
	if err, ok := err.(*DNSError); ok {
		// Show original name passed to lookup, not suffixed one.
		// In general we might have tried many suffixes; showing
		// just one is misleading. See also golang.org/issue/6324.
		err.Name = name
	}
	return
}

// avoidDNS reports whether this is a hostname for which we should not
// use DNS. Currently this includes only .onion, per RFC 7686. See
// golang.org/issue/13705. Does not cover .local names (RFC 6762),
// see golang.org/issue/16739.
func avoidDNS(name string) bool {
	if name == "" {
		return true
	}
	if name[len(name)-1] == '.' {
		name = name[:len(name)-1]
	}
	return stringsHasSuffixFold(name, ".onion")
}

// nameList returns a list of names for sequential DNS queries.
func (conf *dnsConfig) nameList(name string) []string {
	if avoidDNS(name) {
		return nil
	}

	// If name is rooted (trailing dot), try only that name.
	rooted := len(name) > 0 && name[len(name)-1] == '.'
	if rooted {
		return []string{name}
	}

	hasNdots := count(name, '.') >= conf.ndots
	name += "."

	// Build list of search choices.
	names := make([]string, 0, 1+len(conf.search))
	// If name has enough dots, try unsuffixed first.
	if hasNdots {
		names = append(names, name)
	}
	// Try suffixes.
	for _, suffix := range conf.search {
		names = append(names, name+suffix)
	}
	// Try unsuffixed, if not tried first above.
	if !hasNdots {
		names = append(names, name)
	}
	return names
}

// hostLookupOrder specifies the order of LookupHost lookup strategies.
// It is basically a simplified representation of nsswitch.conf.
// "files" means /etc/hosts.
type hostLookupOrder int

const (
	// hostLookupCgo means defer to cgo.
	hostLookupCgo      hostLookupOrder = iota
	hostLookupFilesDNS                 // files first
	hostLookupDNSFiles                 // dns first
	hostLookupFiles                    // only files
	hostLookupDNS                      // only DNS
)

var lookupOrderName = map[hostLookupOrder]string{
	hostLookupCgo:      "cgo",
	hostLookupFilesDNS: "files,dns",
	hostLookupDNSFiles: "dns,files",
	hostLookupFiles:    "files",
	hostLookupDNS:      "dns",
}

func (o hostLookupOrder) String() string {
	if s, ok := lookupOrderName[o]; ok {
		return s
	}
	return "hostLookupOrder=" + itoa(int(o)) + "??"
}

// goLookupHost is the native Go implementation of LookupHost.
// Used only if cgoLookupHost refuses to handle the request
// (that is, only if cgoLookupHost is the stub in cgo_stub.go).
// Normally we let cgo use the C library resolver instead of
// depending on our lookup code, so that Go and C get the same
// answers.
func goLookupHost(ctx context.Context, name string) (addrs []string, err error) {
	return goLookupHostOrder(ctx, name, hostLookupFilesDNS)
}

func goLookupHostOrder(ctx context.Context, name string, order hostLookupOrder) (addrs []string, err error) {
	if order == hostLookupFilesDNS || order == hostLookupFiles {
		// Use entries from /etc/hosts if they match.
		addrs = lookupStaticHost(name)
		if len(addrs) > 0 || order == hostLookupFiles {
			return
		}
	}
	ips, err := goLookupIPOrder(ctx, name, order)
	if err != nil {
		return
	}
	addrs = make([]string, 0, len(ips))
	for _, ip := range ips {
		addrs = append(addrs, ip.String())
	}
	return
}

// lookup entries from /etc/hosts
func goLookupIPFiles(name string) (addrs []IPAddr) {
	for _, haddr := range lookupStaticHost(name) {
		haddr, zone := splitHostZone(haddr)
		if ip := ParseIP(haddr); ip != nil {
			addr := IPAddr{IP: ip, Zone: zone}
			addrs = append(addrs, addr)
		}
	}
	sortByRFC6724(addrs)
	return
}

// goLookupIP is the native Go implementation of LookupIP.
// The libc versions are in cgo_*.go.
func goLookupIP(ctx context.Context, name string) (addrs []IPAddr, err error) {
	return goLookupIPOrder(ctx, name, hostLookupFilesDNS)
}

func goLookupIPOrder(ctx context.Context, name string, order hostLookupOrder) (addrs []IPAddr, err error) {
	if order == hostLookupFilesDNS || order == hostLookupFiles {
		addrs = goLookupIPFiles(name)
		if len(addrs) > 0 || order == hostLookupFiles {
			return addrs, nil
		}
	}
	if !isDomainName(name) {
		return nil, &DNSError{Err: "invalid domain name", Name: name}
	}
	resolvConf.tryUpdate("/etc/resolv.conf")
	resolvConf.mu.RLock()
	conf := resolvConf.dnsConfig
	resolvConf.mu.RUnlock()
	type racer struct {
		fqdn string
		rrs  []dnsRR
		error
	}
	lane := make(chan racer, 1)
	qtypes := [...]uint16{dnsTypeA, dnsTypeAAAA}
	var lastErr error
	for _, fqdn := range conf.nameList(name) {
		for _, qtype := range qtypes {
			go func(qtype uint16) {
				_, rrs, err := tryOneName(ctx, conf, fqdn, qtype)
				lane <- racer{fqdn, rrs, err}
			}(qtype)
		}
		for range qtypes {
			racer := <-lane
			if racer.error != nil {
				// Prefer error for original name.
				if lastErr == nil || racer.fqdn == name+"." {
					lastErr = racer.error
				}
				continue
			}
			addrs = append(addrs, addrRecordList(racer.rrs)...)
		}
		if len(addrs) > 0 {
			break
		}
	}
	if lastErr, ok := lastErr.(*DNSError); ok {
		// Show original name passed to lookup, not suffixed one.
		// In general we might have tried many suffixes; showing
		// just one is misleading. See also golang.org/issue/6324.
		lastErr.Name = name
	}
	sortByRFC6724(addrs)
	if len(addrs) == 0 {
		if order == hostLookupDNSFiles {
			addrs = goLookupIPFiles(name)
		}
		if len(addrs) == 0 && lastErr != nil {
			return nil, lastErr
		}
	}
	return addrs, nil
}

// goLookupCNAME is the native Go implementation of LookupCNAME.
// Used only if cgoLookupCNAME refuses to handle the request
// (that is, only if cgoLookupCNAME is the stub in cgo_stub.go).
// Normally we let cgo use the C library resolver instead of
// depending on our lookup code, so that Go and C get the same
// answers.
func goLookupCNAME(ctx context.Context, name string) (cname string, err error) {
	_, rrs, err := lookup(ctx, name, dnsTypeCNAME)
	if err != nil {
		return
	}
	cname = rrs[0].(*dnsRR_CNAME).Cname
	return
}

// goLookupPTR is the native Go implementation of LookupAddr.
// Used only if cgoLookupPTR refuses to handle the request (that is,
// only if cgoLookupPTR is the stub in cgo_stub.go).
// Normally we let cgo use the C library resolver instead of depending
// on our lookup code, so that Go and C get the same answers.
func goLookupPTR(ctx context.Context, addr string) ([]string, error) {
	names := lookupStaticAddr(addr)
	if len(names) > 0 {
		return names, nil
	}
	arpa, err := reverseaddr(addr)
	if err != nil {
		return nil, err
	}
	_, rrs, err := lookup(ctx, arpa, dnsTypePTR)
	if err != nil {
		return nil, err
	}
	ptrs := make([]string, len(rrs))
	for i, rr := range rrs {
		ptrs[i] = rr.(*dnsRR_PTR).Ptr
	}
	return ptrs, nil
}

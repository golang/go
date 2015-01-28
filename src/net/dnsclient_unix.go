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
	"errors"
	"io"
	"math/rand"
	"os"
	"strconv"
	"sync"
	"time"
)

// A dnsConn represents a DNS transport endpoint.
type dnsConn interface {
	Conn

	// readDNSResponse reads a DNS response message from the DNS
	// transport endpoint and returns the received DNS response
	// message.
	readDNSResponse() (*dnsMsg, error)

	// writeDNSQuery writes a DNS query message to the DNS
	// connection endpoint.
	writeDNSQuery(*dnsMsg) error
}

func (c *UDPConn) readDNSResponse() (*dnsMsg, error) {
	b := make([]byte, 512) // see RFC 1035
	n, err := c.Read(b)
	if err != nil {
		return nil, err
	}
	msg := &dnsMsg{}
	if !msg.Unpack(b[:n]) {
		return nil, errors.New("cannot unmarshal DNS message")
	}
	return msg, nil
}

func (c *UDPConn) writeDNSQuery(msg *dnsMsg) error {
	b, ok := msg.Pack()
	if !ok {
		return errors.New("cannot marshal DNS message")
	}
	if _, err := c.Write(b); err != nil {
		return err
	}
	return nil
}

func (c *TCPConn) readDNSResponse() (*dnsMsg, error) {
	b := make([]byte, 1280) // 1280 is a reasonable initial size for IP over Ethernet, see RFC 4035
	if _, err := io.ReadFull(c, b[:2]); err != nil {
		return nil, err
	}
	l := int(b[0])<<8 | int(b[1])
	if l > len(b) {
		b = make([]byte, l)
	}
	n, err := io.ReadFull(c, b[:l])
	if err != nil {
		return nil, err
	}
	msg := &dnsMsg{}
	if !msg.Unpack(b[:n]) {
		return nil, errors.New("cannot unmarshal DNS message")
	}
	return msg, nil
}

func (c *TCPConn) writeDNSQuery(msg *dnsMsg) error {
	b, ok := msg.Pack()
	if !ok {
		return errors.New("cannot marshal DNS message")
	}
	l := uint16(len(b))
	b = append([]byte{byte(l >> 8), byte(l)}, b...)
	if _, err := c.Write(b); err != nil {
		return err
	}
	return nil
}

func (d *Dialer) dialDNS(network, server string) (dnsConn, error) {
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
	c, err := d.Dial(network, server)
	if err != nil {
		return nil, err
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
func exchange(server, name string, qtype uint16, timeout time.Duration) (*dnsMsg, error) {
	d := Dialer{Timeout: timeout}
	out := dnsMsg{
		dnsMsgHdr: dnsMsgHdr{
			recursion_desired: true,
		},
		question: []dnsQuestion{
			{name, qtype, dnsClassINET},
		},
	}
	for _, network := range []string{"udp", "tcp"} {
		c, err := d.dialDNS(network, server)
		if err != nil {
			return nil, err
		}
		defer c.Close()
		if timeout > 0 {
			c.SetDeadline(time.Now().Add(timeout))
		}
		out.id = uint16(rand.Int()) ^ uint16(time.Now().UnixNano())
		if err := c.writeDNSQuery(&out); err != nil {
			return nil, err
		}
		in, err := c.readDNSResponse()
		if err != nil {
			return nil, err
		}
		if in.id != out.id {
			return nil, errors.New("DNS message ID mismatch")
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
func tryOneName(cfg *dnsConfig, name string, qtype uint16) (string, []dnsRR, error) {
	if len(cfg.servers) == 0 {
		return "", nil, &DNSError{Err: "no DNS servers", Name: name}
	}
	if len(name) >= 256 {
		return "", nil, &DNSError{Err: "DNS name too long", Name: name}
	}
	timeout := time.Duration(cfg.timeout) * time.Second
	var lastErr error
	for i := 0; i < cfg.attempts; i++ {
		for _, server := range cfg.servers {
			server = JoinHostPort(server, "53")
			msg, err := exchange(server, name, qtype, timeout)
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
			cname, addrs, err := answer(name, server, msg, qtype)
			if err == nil || err.(*DNSError).Err == errNoSuchHost.Error() {
				return cname, addrs, err
			}
			lastErr = err
		}
	}
	return "", nil, lastErr
}

func convertRR_A(records []dnsRR) []IP {
	addrs := make([]IP, len(records))
	for i, rr := range records {
		a := rr.(*dnsRR_A).A
		addrs[i] = IPv4(byte(a>>24), byte(a>>16), byte(a>>8), byte(a))
	}
	return addrs
}

func convertRR_AAAA(records []dnsRR) []IP {
	addrs := make([]IP, len(records))
	for i, rr := range records {
		a := make(IP, IPv6len)
		copy(a, rr.(*dnsRR_AAAA).AAAA[:])
		addrs[i] = a
	}
	return addrs
}

// cfg is used for the storage and reparsing of /etc/resolv.conf
var cfg struct {
	// ch is used as a semaphore that only allows one lookup at a time to
	// recheck resolv.conf.  It acts as guard for lastChecked and modTime.
	ch          chan struct{}
	lastChecked time.Time // last time resolv.conf was checked
	modTime     time.Time // time of resolv.conf modification

	mu        sync.RWMutex // protects dnsConfig
	dnsConfig *dnsConfig   // parsed resolv.conf structure used in lookups
}

var onceLoadConfig sync.Once

func initCfg() {
	// Set dnsConfig, modTime, and lastChecked so we don't parse
	// resolv.conf twice the first time.
	cfg.dnsConfig = systemConf().resolv
	if cfg.dnsConfig == nil {
		cfg.dnsConfig = dnsReadConfig("/etc/resolv.conf")
	}

	if fi, err := os.Stat("/etc/resolv.conf"); err == nil {
		cfg.modTime = fi.ModTime()
	}
	cfg.lastChecked = time.Now()

	// Prepare ch so that only one loadConfig may run at once
	cfg.ch = make(chan struct{}, 1)
	cfg.ch <- struct{}{}
}

func loadConfig(resolvConfPath string) {
	onceLoadConfig.Do(initCfg)

	// ensure only one loadConfig at a time checks /etc/resolv.conf
	select {
	case <-cfg.ch:
		defer func() { cfg.ch <- struct{}{} }()
	default:
		return
	}

	now := time.Now()
	if cfg.lastChecked.After(now.Add(-5 * time.Second)) {
		return
	}
	cfg.lastChecked = now

	if fi, err := os.Stat(resolvConfPath); err == nil {
		if fi.ModTime().Equal(cfg.modTime) {
			return
		}
		cfg.modTime = fi.ModTime()
	} else {
		// If modTime wasn't set prior, assume nothing has changed.
		if cfg.modTime.IsZero() {
			return
		}
		cfg.modTime = time.Time{}
	}

	ncfg := dnsReadConfig(resolvConfPath)
	cfg.mu.Lock()
	cfg.dnsConfig = ncfg
	cfg.mu.Unlock()
}

func lookup(name string, qtype uint16) (cname string, rrs []dnsRR, err error) {
	if !isDomainName(name) {
		return name, nil, &DNSError{Err: "invalid domain name", Name: name}
	}

	loadConfig("/etc/resolv.conf")
	cfg.mu.RLock()
	resolv := cfg.dnsConfig
	cfg.mu.RUnlock()

	// If name is rooted (trailing dot) or has enough dots,
	// try it by itself first.
	rooted := len(name) > 0 && name[len(name)-1] == '.'
	if rooted || count(name, '.') >= resolv.ndots {
		rname := name
		if !rooted {
			rname += "."
		}
		// Can try as ordinary name.
		cname, rrs, err = tryOneName(resolv, rname, qtype)
		if rooted || err == nil {
			return
		}
	}

	// Otherwise, try suffixes.
	for _, suffix := range resolv.search {
		rname := name + "." + suffix
		if rname[len(rname)-1] != '.' {
			rname += "."
		}
		cname, rrs, err = tryOneName(resolv, rname, qtype)
		if err == nil {
			return
		}
	}

	// Last ditch effort: try unsuffixed only if we haven't already,
	// that is, name is not rooted and has less than ndots dots.
	if count(name, '.') < resolv.ndots {
		cname, rrs, err = tryOneName(resolv, name+".", qtype)
		if err == nil {
			return
		}
	}

	if e, ok := err.(*DNSError); ok {
		// Show original name passed to lookup, not suffixed one.
		// In general we might have tried many suffixes; showing
		// just one is misleading. See also golang.org/issue/6324.
		e.Name = name
	}
	return
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
	return "hostLookupOrder=" + strconv.Itoa(int(o)) + "??"
}

// goLookupHost is the native Go implementation of LookupHost.
// Used only if cgoLookupHost refuses to handle the request
// (that is, only if cgoLookupHost is the stub in cgo_stub.go).
// Normally we let cgo use the C library resolver instead of
// depending on our lookup code, so that Go and C get the same
// answers.
func goLookupHost(name string) (addrs []string, err error) {
	return goLookupHostOrder(name, hostLookupFilesDNS)
}

func goLookupHostOrder(name string, order hostLookupOrder) (addrs []string, err error) {
	if order == hostLookupFilesDNS || order == hostLookupFiles {
		// Use entries from /etc/hosts if they match.
		addrs = lookupStaticHost(name)
		if len(addrs) > 0 || order == hostLookupFiles {
			return
		}
	}
	ips, err := goLookupIPOrder(name, order)
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
	return
}

// goLookupIP is the native Go implementation of LookupIP.
// Used only if cgoLookupIP refuses to handle the request
// (that is, only if cgoLookupIP is the stub in cgo_stub.go).
func goLookupIP(name string) (addrs []IPAddr, err error) {
	return goLookupIPOrder(name, hostLookupFilesDNS)
}

func goLookupIPOrder(name string, order hostLookupOrder) (addrs []IPAddr, err error) {
	if order == hostLookupFilesDNS || order == hostLookupFiles {
		addrs = goLookupIPFiles(name)
		if len(addrs) > 0 || order == hostLookupFiles {
			return addrs, nil
		}
	}
	type racer struct {
		qtype uint16
		rrs   []dnsRR
		error
	}
	lane := make(chan racer, 1)
	qtypes := [...]uint16{dnsTypeA, dnsTypeAAAA}
	for _, qtype := range qtypes {
		go func(qtype uint16) {
			_, rrs, err := lookup(name, qtype)
			lane <- racer{qtype, rrs, err}
		}(qtype)
	}
	var lastErr error
	for range qtypes {
		racer := <-lane
		if racer.error != nil {
			lastErr = racer.error
			continue
		}
		switch racer.qtype {
		case dnsTypeA:
			for _, ip := range convertRR_A(racer.rrs) {
				addr := IPAddr{IP: ip}
				addrs = append(addrs, addr)
			}
		case dnsTypeAAAA:
			for _, ip := range convertRR_AAAA(racer.rrs) {
				addr := IPAddr{IP: ip}
				addrs = append(addrs, addr)
			}
		}
	}
	if len(addrs) == 0 {
		if lastErr != nil {
			return nil, lastErr
		}
		if order == hostLookupDNSFiles {
			addrs = goLookupIPFiles(name)
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
func goLookupCNAME(name string) (cname string, err error) {
	_, rr, err := lookup(name, dnsTypeCNAME)
	if err != nil {
		return
	}
	cname = rr[0].(*dnsRR_CNAME).Cname
	return
}

// goLookupPTR is the native Go implementation of LookupAddr.
// Used only if cgoLookupPTR refuses to handle the request (that is,
// only if cgoLookupPTR is the stub in cgo_stub.go).
// Normally we let cgo use the C library resolver instead of depending
// on our lookup code, so that Go and C get the same answers.
func goLookupPTR(addr string) ([]string, error) {
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

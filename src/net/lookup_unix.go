// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package net

import (
	"context"
	"internal/bytealg"
	"sync"
	"syscall"

	"golang.org/x/net/dns/dnsmessage"
)

var onceReadProtocols sync.Once

// readProtocols loads contents of /etc/protocols into protocols map
// for quick access.
func readProtocols() {
	file, err := open("/etc/protocols")
	if err != nil {
		return
	}
	defer file.close()

	for line, ok := file.readLine(); ok; line, ok = file.readLine() {
		// tcp    6   TCP    # transmission control protocol
		if i := bytealg.IndexByteString(line, '#'); i >= 0 {
			line = line[0:i]
		}
		f := getFields(line)
		if len(f) < 2 {
			continue
		}
		if proto, _, ok := dtoi(f[1]); ok {
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
}

// lookupProtocol looks up IP protocol name in /etc/protocols and
// returns correspondent protocol number.
func lookupProtocol(_ context.Context, name string) (int, error) {
	onceReadProtocols.Do(readProtocols)
	return lookupProtocolMap(name)
}

func (r *Resolver) dial(ctx context.Context, network, server string) (Conn, error) {
	// Calling Dial here is scary -- we have to be sure not to
	// dial a name that will require a DNS lookup, or Dial will
	// call back here to translate it. The DNS config parser has
	// already checked that all the cfg.servers are IP
	// addresses, which Dial will use without a DNS lookup.
	var c Conn
	var err error
	if r != nil && r.Dial != nil {
		c, err = r.Dial(ctx, network, server)
	} else {
		var d Dialer
		c, err = d.DialContext(ctx, network, server)
	}
	if err != nil {
		return nil, mapErr(err)
	}
	return c, nil
}

func (r *Resolver) lookupHost(ctx context.Context, host string) (addrs []string, err error) {
	order := systemConf().hostLookupOrder(r, host)
	if !r.preferGo() && order == hostLookupCgo {
		if addrs, err, ok := cgoLookupHost(ctx, host); ok {
			return addrs, err
		}
		// cgo not available (or netgo); fall back to Go's DNS resolver
		order = hostLookupFilesDNS
	}
	return r.goLookupHostOrder(ctx, host, order)
}

func (r *Resolver) lookupIP(ctx context.Context, network, host string) (addrs []IPAddr, err error) {
	if r.preferGo() {
		return r.goLookupIP(ctx, network, host)
	}
	order := systemConf().hostLookupOrder(r, host)
	if order == hostLookupCgo {
		if addrs, err, ok := cgoLookupIP(ctx, network, host); ok {
			return addrs, err
		}
		// cgo not available (or netgo); fall back to Go's DNS resolver
		order = hostLookupFilesDNS
	}
	ips, _, err := r.goLookupIPCNAMEOrder(ctx, network, host, order)
	return ips, err
}

func (r *Resolver) lookupPort(ctx context.Context, network, service string) (int, error) {
	if !r.preferGo() && systemConf().canUseCgo() {
		if port, err, ok := cgoLookupPort(ctx, network, service); ok {
			if err != nil {
				// Issue 18213: if cgo fails, first check to see whether we
				// have the answer baked-in to the net package.
				if port, err := goLookupPort(network, service); err == nil {
					return port, nil
				}
			}
			return port, err
		}
	}
	return goLookupPort(network, service)
}

func (r *Resolver) lookupCNAME(ctx context.Context, name string) (string, error) {
	if !r.preferGo() && systemConf().canUseCgo() {
		if cname, err, ok := cgoLookupCNAME(ctx, name); ok {
			return cname, err
		}
	}
	return r.goLookupCNAME(ctx, name)
}

func (r *Resolver) lookupSRV(ctx context.Context, service, proto, name string) (string, []*SRV, error) {
	var target string
	if service == "" && proto == "" {
		target = name
	} else {
		target = "_" + service + "._" + proto + "." + name
	}
	p, server, err := r.lookup(ctx, target, dnsmessage.TypeSRV)
	if err != nil {
		return "", nil, err
	}
	var srvs []*SRV
	var cname dnsmessage.Name
	for {
		h, err := p.AnswerHeader()
		if err == dnsmessage.ErrSectionDone {
			break
		}
		if err != nil {
			return "", nil, &DNSError{
				Err:    "cannot unmarshal DNS message",
				Name:   name,
				Server: server,
			}
		}
		if h.Type != dnsmessage.TypeSRV {
			if err := p.SkipAnswer(); err != nil {
				return "", nil, &DNSError{
					Err:    "cannot unmarshal DNS message",
					Name:   name,
					Server: server,
				}
			}
			continue
		}
		if cname.Length == 0 && h.Name.Length != 0 {
			cname = h.Name
		}
		srv, err := p.SRVResource()
		if err != nil {
			return "", nil, &DNSError{
				Err:    "cannot unmarshal DNS message",
				Name:   name,
				Server: server,
			}
		}
		srvs = append(srvs, &SRV{Target: srv.Target.String(), Port: srv.Port, Priority: srv.Priority, Weight: srv.Weight})
	}
	byPriorityWeight(srvs).sort()
	return cname.String(), srvs, nil
}

func (r *Resolver) lookupMX(ctx context.Context, name string) ([]*MX, error) {
	p, server, err := r.lookup(ctx, name, dnsmessage.TypeMX)
	if err != nil {
		return nil, err
	}
	var mxs []*MX
	for {
		h, err := p.AnswerHeader()
		if err == dnsmessage.ErrSectionDone {
			break
		}
		if err != nil {
			return nil, &DNSError{
				Err:    "cannot unmarshal DNS message",
				Name:   name,
				Server: server,
			}
		}
		if h.Type != dnsmessage.TypeMX {
			if err := p.SkipAnswer(); err != nil {
				return nil, &DNSError{
					Err:    "cannot unmarshal DNS message",
					Name:   name,
					Server: server,
				}
			}
			continue
		}
		mx, err := p.MXResource()
		if err != nil {
			return nil, &DNSError{
				Err:    "cannot unmarshal DNS message",
				Name:   name,
				Server: server,
			}
		}
		mxs = append(mxs, &MX{Host: mx.MX.String(), Pref: mx.Pref})

	}
	byPref(mxs).sort()
	return mxs, nil
}

func (r *Resolver) lookupNS(ctx context.Context, name string) ([]*NS, error) {
	p, server, err := r.lookup(ctx, name, dnsmessage.TypeNS)
	if err != nil {
		return nil, err
	}
	var nss []*NS
	for {
		h, err := p.AnswerHeader()
		if err == dnsmessage.ErrSectionDone {
			break
		}
		if err != nil {
			return nil, &DNSError{
				Err:    "cannot unmarshal DNS message",
				Name:   name,
				Server: server,
			}
		}
		if h.Type != dnsmessage.TypeNS {
			if err := p.SkipAnswer(); err != nil {
				return nil, &DNSError{
					Err:    "cannot unmarshal DNS message",
					Name:   name,
					Server: server,
				}
			}
			continue
		}
		ns, err := p.NSResource()
		if err != nil {
			return nil, &DNSError{
				Err:    "cannot unmarshal DNS message",
				Name:   name,
				Server: server,
			}
		}
		nss = append(nss, &NS{Host: ns.NS.String()})
	}
	return nss, nil
}

func (r *Resolver) lookupTXT(ctx context.Context, name string) ([]string, error) {
	p, server, err := r.lookup(ctx, name, dnsmessage.TypeTXT)
	if err != nil {
		return nil, err
	}
	var txts []string
	for {
		h, err := p.AnswerHeader()
		if err == dnsmessage.ErrSectionDone {
			break
		}
		if err != nil {
			return nil, &DNSError{
				Err:    "cannot unmarshal DNS message",
				Name:   name,
				Server: server,
			}
		}
		if h.Type != dnsmessage.TypeTXT {
			if err := p.SkipAnswer(); err != nil {
				return nil, &DNSError{
					Err:    "cannot unmarshal DNS message",
					Name:   name,
					Server: server,
				}
			}
			continue
		}
		txt, err := p.TXTResource()
		if err != nil {
			return nil, &DNSError{
				Err:    "cannot unmarshal DNS message",
				Name:   name,
				Server: server,
			}
		}
		// Multiple strings in one TXT record need to be
		// concatenated without separator to be consistent
		// with previous Go resolver.
		n := 0
		for _, s := range txt.TXT {
			n += len(s)
		}
		txtJoin := make([]byte, 0, n)
		for _, s := range txt.TXT {
			txtJoin = append(txtJoin, s...)
		}
		if len(txts) == 0 {
			txts = make([]string, 0, 1)
		}
		txts = append(txts, string(txtJoin))
	}
	return txts, nil
}

func (r *Resolver) lookupAddr(ctx context.Context, addr string) ([]string, error) {
	if !r.preferGo() && systemConf().canUseCgo() {
		if ptrs, err, ok := cgoLookupPTR(ctx, addr); ok {
			return ptrs, err
		}
	}
	return r.goLookupPTR(ctx, addr)
}

// concurrentThreadsLimit returns the number of threads we permit to
// run concurrently doing DNS lookups via cgo. A DNS lookup may use a
// file descriptor so we limit this to less than the number of
// permitted open files. On some systems, notably Darwin, if
// getaddrinfo is unable to open a file descriptor it simply returns
// EAI_NONAME rather than a useful error. Limiting the number of
// concurrent getaddrinfo calls to less than the permitted number of
// file descriptors makes that error less likely. We don't bother to
// apply the same limit to DNS lookups run directly from Go, because
// there we will return a meaningful "too many open files" error.
func concurrentThreadsLimit() int {
	var rlim syscall.Rlimit
	if err := syscall.Getrlimit(syscall.RLIMIT_NOFILE, &rlim); err != nil {
		return 500
	}
	r := int(rlim.Cur)
	if r > 500 {
		r = 500
	} else if r > 30 {
		r -= 30
	}
	return r
}

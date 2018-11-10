// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package net

import (
	"context"
	"sync"
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
		if i := byteIndex(line, '#'); i >= 0 {
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

func (r *Resolver) dial(ctx context.Context, network, server string) (dnsConn, error) {
	// Calling Dial here is scary -- we have to be sure not to
	// dial a name that will require a DNS lookup, or Dial will
	// call back here to translate it. The DNS config parser has
	// already checked that all the cfg.servers are IP
	// addresses, which Dial will use without a DNS lookup.
	var c Conn
	var err error
	if r.Dial != nil {
		c, err = r.Dial(ctx, network, server)
	} else {
		var d Dialer
		c, err = d.DialContext(ctx, network, server)
	}
	if err != nil {
		return nil, mapErr(err)
	}
	if _, ok := c.(PacketConn); ok {
		return &dnsPacketConn{c}, nil
	}
	return &dnsStreamConn{c}, nil
}

func (r *Resolver) lookupHost(ctx context.Context, host string) (addrs []string, err error) {
	order := systemConf().hostLookupOrder(host)
	if !r.PreferGo && order == hostLookupCgo {
		if addrs, err, ok := cgoLookupHost(ctx, host); ok {
			return addrs, err
		}
		// cgo not available (or netgo); fall back to Go's DNS resolver
		order = hostLookupFilesDNS
	}
	return r.goLookupHostOrder(ctx, host, order)
}

func (r *Resolver) lookupIP(ctx context.Context, host string) (addrs []IPAddr, err error) {
	if r.PreferGo {
		return r.goLookupIP(ctx, host)
	}
	order := systemConf().hostLookupOrder(host)
	if order == hostLookupCgo {
		if addrs, err, ok := cgoLookupIP(ctx, host); ok {
			return addrs, err
		}
		// cgo not available (or netgo); fall back to Go's DNS resolver
		order = hostLookupFilesDNS
	}
	addrs, _, err = r.goLookupIPCNAMEOrder(ctx, host, order)
	return
}

func (r *Resolver) lookupPort(ctx context.Context, network, service string) (int, error) {
	if !r.PreferGo && systemConf().canUseCgo() {
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
	if !r.PreferGo && systemConf().canUseCgo() {
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
	cname, rrs, err := r.lookup(ctx, target, dnsTypeSRV)
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

func (r *Resolver) lookupMX(ctx context.Context, name string) ([]*MX, error) {
	_, rrs, err := r.lookup(ctx, name, dnsTypeMX)
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

func (r *Resolver) lookupNS(ctx context.Context, name string) ([]*NS, error) {
	_, rrs, err := r.lookup(ctx, name, dnsTypeNS)
	if err != nil {
		return nil, err
	}
	nss := make([]*NS, len(rrs))
	for i, rr := range rrs {
		nss[i] = &NS{Host: rr.(*dnsRR_NS).Ns}
	}
	return nss, nil
}

func (r *Resolver) lookupTXT(ctx context.Context, name string) ([]string, error) {
	_, rrs, err := r.lookup(ctx, name, dnsTypeTXT)
	if err != nil {
		return nil, err
	}
	txts := make([]string, len(rrs))
	for i, rr := range rrs {
		txts[i] = rr.(*dnsRR_TXT).Txt
	}
	return txts, nil
}

func (r *Resolver) lookupAddr(ctx context.Context, addr string) ([]string, error) {
	if !r.PreferGo && systemConf().canUseCgo() {
		if ptrs, err, ok := cgoLookupPTR(ctx, addr); ok {
			return ptrs, err
		}
	}
	return r.goLookupPTR(ctx, addr)
}

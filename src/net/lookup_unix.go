// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || js || wasip1

package net

import (
	"context"
	"internal/bytealg"
	"sync"
)

// readProtocolsOnce loads contents of /etc/protocols into protocols map
// for quick access.
var readProtocolsOnce = sync.OnceFunc(func() {
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
})

// lookupProtocol looks up IP protocol name in /etc/protocols and
// returns correspondent protocol number.
func lookupProtocol(_ context.Context, name string) (int, error) {
	readProtocolsOnce()
	return lookupProtocolMap(name)
}

func (r *Resolver) lookupHost(ctx context.Context, host string) (addrs []string, err error) {
	order, conf := systemConf().hostLookupOrder(r, host)
	if order == hostLookupCgo {
		return cgoLookupHost(ctx, host)
	}
	return r.goLookupHostOrder(ctx, host, order, conf)
}

func (r *Resolver) lookupIP(ctx context.Context, network, host string) (addrs []IPAddr, err error) {
	order, conf := systemConf().hostLookupOrder(r, host)
	if order == hostLookupCgo {
		return cgoLookupIP(ctx, network, host)
	}
	ips, _, err := r.goLookupIPCNAMEOrder(ctx, network, host, order, conf)
	return ips, err
}

func (r *Resolver) lookupPort(ctx context.Context, network, service string) (int, error) {
	// Port lookup is not a DNS operation.
	// Prefer the cgo resolver if possible.
	if !systemConf().mustUseGoResolver(r) {
		port, err := cgoLookupPort(ctx, network, service)
		if err != nil {
			// Issue 18213: if cgo fails, first check to see whether we
			// have the answer baked-in to the net package.
			if port, err := goLookupPort(network, service); err == nil {
				return port, nil
			}
		}
		return port, err
	}
	return goLookupPort(network, service)
}

func (r *Resolver) lookupCNAME(ctx context.Context, name string) (string, error) {
	order, conf := systemConf().hostLookupOrder(r, name)
	if order == hostLookupCgo {
		if cname, err, ok := cgoLookupCNAME(ctx, name); ok {
			return cname, err
		}
	}
	return r.goLookupCNAME(ctx, name, order, conf)
}

func (r *Resolver) lookupSRV(ctx context.Context, service, proto, name string) (string, []*SRV, error) {
	return r.goLookupSRV(ctx, service, proto, name)
}

func (r *Resolver) lookupMX(ctx context.Context, name string) ([]*MX, error) {
	return r.goLookupMX(ctx, name)
}

func (r *Resolver) lookupNS(ctx context.Context, name string) ([]*NS, error) {
	return r.goLookupNS(ctx, name)
}

func (r *Resolver) lookupTXT(ctx context.Context, name string) ([]string, error) {
	return r.goLookupTXT(ctx, name)
}

func (r *Resolver) lookupAddr(ctx context.Context, addr string) ([]string, error) {
	order, conf := systemConf().addrLookupOrder(r, addr)
	if order == hostLookupCgo {
		return cgoLookupPTR(ctx, addr)
	}
	return r.goLookupPTR(ctx, addr, order, conf)
}

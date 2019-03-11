// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix dragonfly freebsd linux netbsd openbsd solaris

package net

import (
	"context"
)

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
		return r.goLookupIP(ctx, host)
	}
	order := systemConf().hostLookupOrder(r, host)
	if order == hostLookupCgo {
		if addrs, err, ok := cgoLookupIP(ctx, network, host); ok {
			return addrs, err
		}
		// cgo not available (or netgo); fall back to Go's DNS resolver
		order = hostLookupFilesDNS
	}
	ips, _, err := r.goLookupIPCNAMEOrder(ctx, host, order)
	return ips, err
}

func (r *Resolver) lookupCNAME(ctx context.Context, name string) (string, error) {
	if !r.preferGo() && systemConf().canUseCgo() {
		if cname, err, ok := cgoLookupCNAME(ctx, name); ok {
			return cname, err
		}
	}
	return r.goLookupCNAME(ctx, name)
}

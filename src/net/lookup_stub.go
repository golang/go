// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build nacl

package net

import (
	"context"
	"syscall"
)

func lookupProtocol(ctx context.Context, name string) (proto int, err error) {
	return 0, syscall.ENOPROTOOPT
}

func lookupHost(ctx context.Context, host string) (addrs []string, err error) {
	return nil, syscall.ENOPROTOOPT
}

func lookupIP(ctx context.Context, host string) (addrs []IPAddr, err error) {
	return nil, syscall.ENOPROTOOPT
}

func lookupPort(ctx context.Context, network, service string) (port int, err error) {
	return 0, syscall.ENOPROTOOPT
}

func lookupCNAME(ctx context.Context, name string) (cname string, err error) {
	return "", syscall.ENOPROTOOPT
}

func lookupSRV(ctx context.Context, service, proto, name string) (cname string, srvs []*SRV, err error) {
	return "", nil, syscall.ENOPROTOOPT
}

func lookupMX(ctx context.Context, name string) (mxs []*MX, err error) {
	return nil, syscall.ENOPROTOOPT
}

func lookupNS(ctx context.Context, name string) (nss []*NS, err error) {
	return nil, syscall.ENOPROTOOPT
}

func lookupTXT(ctx context.Context, name string) (txts []string, err error) {
	return nil, syscall.ENOPROTOOPT
}

func lookupAddr(ctx context.Context, addr string) (ptrs []string, err error) {
	return nil, syscall.ENOPROTOOPT
}

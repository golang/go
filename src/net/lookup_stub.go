// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build nacl

package net

import "syscall"

func lookupProtocol(name string) (proto int, err error) {
	return 0, syscall.ENOPROTOOPT
}

func lookupHost(host string) (addrs []string, err error) {
	return nil, syscall.ENOPROTOOPT
}

func lookupIP(host string) (ips []IP, err error) {
	return nil, syscall.ENOPROTOOPT
}

func lookupPort(network, service string) (port int, err error) {
	return 0, syscall.ENOPROTOOPT
}

func lookupCNAME(name string) (cname string, err error) {
	return "", syscall.ENOPROTOOPT
}

func lookupSRV(service, proto, name string) (cname string, srvs []*SRV, err error) {
	return "", nil, syscall.ENOPROTOOPT
}

func lookupMX(name string) (mxs []*MX, err error) {
	return nil, syscall.ENOPROTOOPT
}

func lookupNS(name string) (nss []*NS, err error) {
	return nil, syscall.ENOPROTOOPT
}

func lookupTXT(name string) (txts []string, err error) {
	return nil, syscall.ENOPROTOOPT
}

func lookupAddr(addr string) (ptrs []string, err error) {
	return nil, syscall.ENOPROTOOPT
}

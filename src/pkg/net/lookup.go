// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"os"
)

// LookupHost looks up the given host using the local resolver.
// It returns an array of that host's addresses.
func LookupHost(host string) (addrs []string, err os.Error) {
	addrs, err, ok := cgoLookupHost(host)
	if !ok {
		addrs, err = goLookupHost(host)
	}
	return
}

// LookupIP looks up host using the local resolver.
// It returns an array of that host's IPv4 and IPv6 addresses.
func LookupIP(host string) (addrs []IP, err os.Error) {
	addrs, err, ok := cgoLookupIP(host)
	if !ok {
		addrs, err = goLookupIP(host)
	}
	return
}

// LookupPort looks up the port for the given network and service.
func LookupPort(network, service string) (port int, err os.Error) {
	port, err, ok := cgoLookupPort(network, service)
	if !ok {
		port, err = goLookupPort(network, service)
	}
	return
}

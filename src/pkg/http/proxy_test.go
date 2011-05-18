// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"os"
	"testing"
)

// TODO(mattn):
//	test ProxyAuth

var UseProxyTests = []struct {
	host  string
	match bool
}{
	// Never proxy localhost:
	{"localhost:80", false},
	{"127.0.0.1", false},
	{"127.0.0.2", false},
	{"[::1]", false},
	{"[::2]", true}, // not a loopback address

	{"barbaz.net", false},     // match as .barbaz.net
	{"foobar.com", false},     // have a port but match 
	{"foofoobar.com", true},   // not match as a part of foobar.com
	{"baz.com", true},         // not match as a part of barbaz.com
	{"localhost.net", true},   // not match as suffix of address
	{"local.localhost", true}, // not match as prefix as address
	{"barbarbaz.net", true},   // not match because NO_PROXY have a '.'
	{"www.foobar.com", true},  // not match because NO_PROXY is not .foobar.com
}

func TestUseProxy(t *testing.T) {
	oldenv := os.Getenv("NO_PROXY")
	defer os.Setenv("NO_PROXY", oldenv)

	no_proxy := "foobar.com, .barbaz.net"
	os.Setenv("NO_PROXY", no_proxy)

	for _, test := range UseProxyTests {
		if useProxy(test.host+":80") != test.match {
			t.Errorf("useProxy(%v) = %v, want %v", test.host, !test.match, test.match)
		}
	}
}

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
	{"localhost", false},      // match completely
	{"barbaz.net", false},     // match as .barbaz.net
	{"foobar.com:443", false}, // have a port but match 
	{"foofoobar.com", true},   // not match as a part of foobar.com
	{"baz.com", true},         // not match as a part of barbaz.com
	{"localhost.net", true},   // not match as suffix of address
	{"local.localhost", true}, // not match as prefix as address
	{"barbarbaz.net", true},   // not match because NO_PROXY have a '.'
	{"www.foobar.com", true},  // not match because NO_PROXY is not .foobar.com
}

func TestUseProxy(t *testing.T) {
	oldenv := os.Getenv("NO_PROXY")
	no_proxy := "foobar.com, .barbaz.net   , localhost"
	os.Setenv("NO_PROXY", no_proxy)
	defer os.Setenv("NO_PROXY", oldenv)

	tr := &Transport{}

	for _, test := range UseProxyTests {
		if tr.useProxy(test.host) != test.match {
			if test.match {
				t.Errorf("useProxy(%v) = %v, want %v", test.host, !test.match, test.match)
			} else {
				t.Errorf("not expected: '%s' shouldn't match as '%s'", test.host, no_proxy)
			}
		}
	}
}

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

var MatchNoProxyTests = []struct {
	host  string
	match bool
}{
	{"localhost", true},        // match completely
	{"barbaz.net", true},       // match as .barbaz.net
	{"foobar.com:443", true},   // have a port but match 
	{"foofoobar.com", false},   // not match as a part of foobar.com
	{"baz.com", false},         // not match as a part of barbaz.com
	{"localhost.net", false},   // not match as suffix of address
	{"local.localhost", false}, // not match as prefix as address
	{"barbarbaz.net", false},   // not match because NO_PROXY have a '.'
	{"www.foobar.com", false},  // not match because NO_PROXY is not .foobar.com
}

func TestMatchNoProxy(t *testing.T) {
	oldenv := os.Getenv("NO_PROXY")
	no_proxy := "foobar.com, .barbaz.net   , localhost"
	os.Setenv("NO_PROXY", no_proxy)
	defer os.Setenv("NO_PROXY", oldenv)

	for _, test := range MatchNoProxyTests {
		if matchNoProxy(test.host) != test.match {
			if test.match {
				t.Errorf("matchNoProxy(%v) = %v, want %v", test.host, !test.match, test.match)
			} else {
				t.Errorf("not expected: '%s' shouldn't match as '%s'", test.host, no_proxy)
			}
		}
	}
}

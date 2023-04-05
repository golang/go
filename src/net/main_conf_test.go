// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !js && !plan9 && !wasip1

package net

import "testing"

// forceGoDNS forces the resolver configuration to use the pure Go resolver
// and returns a fixup function to restore the old settings.
func forceGoDNS() func() {
	c := systemConf()
	oldResolver := c.resolver
	fixup := func() {
		c.resolver = oldResolver
	}
	c.resolver = resolverGo
	return fixup
}

// forceCgoDNS forces the resolver configuration to use the cgo resolver
// and returns a fixup function to restore the old settings.
// (On non-Unix systems forceCgoDNS returns nil.)
func forceCgoDNS() func() {
	c := systemConf()
	oldResolver := c.resolver
	fixup := func() {
		c.resolver = oldResolver
	}
	c.resolver = resolverCgo
	return fixup
}

func TestForceCgoDNS(t *testing.T) {
	defer forceCgoDNS()()
	order, _ := systemConf().hostLookupOrder(nil, "go.dev")
	if order != hostLookupCgo {
		t.Fatalf("hostLookupOrder returned: %v, want cgo", order)
	}
}

func TestForceGoDNS(t *testing.T) {
	defer forceGoDNS()()
	order, _ := systemConf().hostLookupOrder(nil, "go.dev")
	if !(order == hostLookupFiles || order == hostLookupFilesDNS ||
		order == hostLookupDNSFiles || order == hostLookupDNS) {
		t.Fatalf("hostLookupOrder returned: %v, want go resolver order", order)
	}
}

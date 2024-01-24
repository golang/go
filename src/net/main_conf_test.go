// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9

package net

import "testing"

// forceGoDNS forces the resolver configuration to use the pure Go resolver
// and returns a fixup function to restore the old settings.
func forceGoDNS() func() {
	c := systemConf()
	oldGo := c.netGo
	oldCgo := c.netCgo
	fixup := func() {
		c.netGo = oldGo
		c.netCgo = oldCgo
	}
	c.netGo = true
	c.netCgo = false
	return fixup
}

// forceCgoDNS forces the resolver configuration to use the cgo resolver
// and returns a fixup function to restore the old settings.
// (On non-Unix systems forceCgoDNS returns nil.)
func forceCgoDNS() func() {
	c := systemConf()
	oldGo := c.netGo
	oldCgo := c.netCgo
	fixup := func() {
		c.netGo = oldGo
		c.netCgo = oldCgo
	}
	c.netGo = false
	c.netCgo = true
	return fixup
}

func TestForceCgoDNS(t *testing.T) {
	if !cgoAvailable {
		t.Skip("cgo resolver not available")
	}
	defer forceCgoDNS()()
	order, _ := systemConf().hostLookupOrder(nil, "go.dev")
	if order != hostLookupCgo {
		t.Fatalf("hostLookupOrder returned: %v, want cgo", order)
	}
}

func TestForceGoDNS(t *testing.T) {
	defer forceGoDNS()()
	order, _ := systemConf().hostLookupOrder(nil, "go.dev")
	if order == hostLookupCgo {
		t.Fatalf("hostLookupOrder returned: %v, want go resolver order", order)
	}
}

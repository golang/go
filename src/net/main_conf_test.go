// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"context"
	"runtime"
	"testing"
)

func allResolvers(t *testing.T, f func(t *testing.T)) {
	t.Run("default resolver", f)
	t.Run("forced go resolver", func { t ->
		// On plan9 the forceGoDNS might not force the go resolver, currently
		// it is only forced when the Resolver.Dial field is populated.
		// See conf.go mustUseGoResolver.
		defer forceGoDNS()()
		f(t)
	})
	t.Run("forced cgo resolver", func { t ->
		defer forceCgoDNS()()
		f(t)
	})
}

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
	order, _ = systemConf().addrLookupOrder(nil, "192.0.2.1")
	if order != hostLookupCgo {
		t.Fatalf("addrLookupOrder returned: %v, want cgo", order)
	}
	if systemConf().mustUseGoResolver(nil) {
		t.Fatal("mustUseGoResolver = true, want false")
	}
}

func TestForceGoDNS(t *testing.T) {
	var resolver *Resolver
	if runtime.GOOS == "plan9" {
		resolver = &Resolver{
			Dial: func(_ context.Context, _, _ string) (Conn, error) {
				panic("unreachable")
			},
		}
	}
	defer forceGoDNS()()
	order, _ := systemConf().hostLookupOrder(resolver, "go.dev")
	if order == hostLookupCgo {
		t.Fatalf("hostLookupOrder returned: %v, want go resolver order", order)
	}
	order, _ = systemConf().addrLookupOrder(resolver, "192.0.2.1")
	if order == hostLookupCgo {
		t.Fatalf("addrLookupOrder returned: %v, want go resolver order", order)
	}
	if !systemConf().mustUseGoResolver(resolver) {
		t.Fatal("mustUseGoResolver = false, want true")
	}
}

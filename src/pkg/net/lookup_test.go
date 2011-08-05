// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO It would be nice to use a mock DNS server, to eliminate
// external dependencies.

package net

import (
	"runtime"
	"testing"
)

var avoidMacFirewall = runtime.GOOS == "darwin"

func TestGoogleSRV(t *testing.T) {
	if testing.Short() || avoidMacFirewall {
		t.Logf("skipping test to avoid external network")
		return
	}
	_, addrs, err := LookupSRV("xmpp-server", "tcp", "google.com")
	if err != nil {
		t.Errorf("failed: %s", err)
	}
	if len(addrs) == 0 {
		t.Errorf("no results")
	}
}

func TestGmailMX(t *testing.T) {
	if testing.Short() || avoidMacFirewall {
		t.Logf("skipping test to avoid external network")
		return
	}
	mx, err := LookupMX("gmail.com")
	if err != nil {
		t.Errorf("failed: %s", err)
	}
	if len(mx) == 0 {
		t.Errorf("no results")
	}
}

func TestGoogleDNSAddr(t *testing.T) {
	if testing.Short() || avoidMacFirewall {
		t.Logf("skipping test to avoid external network")
		return
	}
	names, err := LookupAddr("8.8.8.8")
	if err != nil {
		t.Errorf("failed: %s", err)
	}
	if len(names) == 0 {
		t.Errorf("no results")
	}
}

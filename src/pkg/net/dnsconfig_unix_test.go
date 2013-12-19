// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd

package net

import "testing"

func TestDNSReadConfig(t *testing.T) {
	dnsConfig, err := dnsReadConfig("testdata/resolv.conf")
	if err != nil {
		t.Fatal(err)
	}

	if len(dnsConfig.servers) != 1 {
		t.Errorf("len(dnsConfig.servers) = %d; want %d", len(dnsConfig.servers), 1)
	}
	if dnsConfig.servers[0] != "[192.168.1.1]" {
		t.Errorf("dnsConfig.servers[0] = %s; want %s", dnsConfig.servers[0], "[192.168.1.1]")
	}

	if len(dnsConfig.search) != 1 {
		t.Errorf("len(dnsConfig.search) = %d; want %d", len(dnsConfig.search), 1)
	}
	if dnsConfig.search[0] != "Home" {
		t.Errorf("dnsConfig.search[0] = %s; want %s", dnsConfig.search[0], "Home")
	}

	if dnsConfig.ndots != 5 {
		t.Errorf("dnsConfig.ndots = %d; want %d", dnsConfig.ndots, 5)
	}

	if dnsConfig.timeout != 10 {
		t.Errorf("dnsConfig.timeout = %d; want %d", dnsConfig.timeout, 10)
	}

	if dnsConfig.attempts != 3 {
		t.Errorf("dnsConfig.attempts = %d; want %d", dnsConfig.attempts, 3)
	}

	if dnsConfig.rotate != true {
		t.Errorf("dnsConfig.rotate = %t; want %t", dnsConfig.rotate, true)
	}
}

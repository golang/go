// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !netgo,!cgo
// +build darwin

package net

import (
	"context"
	"strings"
	"testing"
)

func TestPseudoCgoLookupHost(t *testing.T) {
	addrs, err, ok := cgoLookupHost(context.Background(), "google.com")
	t.Logf("cgoLookupHost google.com: %v %v %v", addrs, err, ok)
	if !ok {
		t.Fatal("cgoLookupHost ok=false")
	}
	if err != nil {
		t.Fatalf("cgoLookupHost: %v", err)
	}
	// cgoLookupHost need not return IPv4 before IPv6 in general,
	// but for the current implementation it does.
	// If that changes, this test will need updating.
	if len(addrs) < 1 || strings.Count(addrs[0], ".") != 3 || !strings.Contains(addrs[len(addrs)-1], "::") {
		t.Fatalf("cgoLookupHost google.com = %v, want IPv4 and IPv6", addrs)
	}
}

func TestPseudoCgoLookupIP(t *testing.T) {
	ips, err, ok := cgoLookupIP(context.Background(), "ip", "google.com")
	t.Logf("cgoLookupIP google.com: %v %v %v", ips, err, ok)
	if !ok {
		t.Fatal("cgoLookupIP ok=false")
	}
	if err != nil {
		t.Fatalf("cgoLookupIP: %v", err)
	}
	// cgoLookupIP need not return IPv4 before IPv6 in general,
	// but for the current implementation it does.
	// If that changes, this test will need updating.
	if len(ips) < 1 || len(ips[0].IP) != 4 || len(ips[len(ips)-1].IP) != 16 {
		t.Fatalf("cgoLookupIP google.com = %v, want IPv4 and IPv6", ips)
	}
}

func TestPseudoCgoLookupCNAME(t *testing.T) {
	t.Skip("res_search on macOS hangs in TypeCNAME queries (even in plain C programs)")

	cname, err, ok := cgoLookupCNAME(context.Background(), "redirect.swtch.com")
	t.Logf("cgoLookupCNAME redirect.swtch.com: %v %v %v", cname, err, ok)
	if !ok {
		t.Fatal("cgoLookupCNAME ok=false")
	}
	if err != nil {
		t.Fatalf("cgoLookupCNAME: %v", err)
	}
	if !strings.HasSuffix(cname, ".com") {
		t.Fatalf("cgoLookupCNAME redirect.swtch.com = %v, want *.com", cname)
	}
}

func TestPseudoCgoLookupPTR(t *testing.T) {
	t.Skip("res_search on macOS does not support TypePTR")

	ptrs, err, ok := cgoLookupPTR(context.Background(), "8.8.8.8")
	t.Logf("cgoLookupPTR 8.8.8.8: %v %v %v", ptrs, err, ok)
	if !ok {
		t.Fatal("cgoLookupPTR ok=false")
	}
	if err != nil {
		t.Fatalf("cgoLookupPTR: %v", err)
	}
	if len(ptrs) < 1 || ptrs[0] != "google-public-dns-a.google.com" {
		t.Fatalf("cgoLookupPTR = %v, want google-public-dns-a.google.com", ptrs)
	}
}

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO It would be nice to use a mock DNS server, to eliminate
// external dependencies.

package net

import (
	"flag"
	"strings"
	"testing"
)

var testExternal = flag.Bool("external", true, "allow use of external networks during long test")

func TestGoogleSRV(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}
	_, addrs, err := LookupSRV("xmpp-server", "tcp", "google.com")
	if err != nil {
		t.Errorf("failed: %s", err)
	}
	if len(addrs) == 0 {
		t.Errorf("no results")
	}

	// Non-standard back door.
	_, addrs, err = LookupSRV("", "", "_xmpp-server._tcp.google.com")
	if err != nil {
		t.Errorf("back door failed: %s", err)
	}
	if len(addrs) == 0 {
		t.Errorf("back door no results")
	}
}

func TestGmailMX(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}
	mx, err := LookupMX("gmail.com")
	if err != nil {
		t.Errorf("failed: %s", err)
	}
	if len(mx) == 0 {
		t.Errorf("no results")
	}
}

func TestGmailNS(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}
	ns, err := LookupNS("gmail.com")
	if err != nil {
		t.Errorf("failed: %s", err)
	}
	if len(ns) == 0 {
		t.Errorf("no results")
	}
}

func TestGmailTXT(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}
	txt, err := LookupTXT("gmail.com")
	if err != nil {
		t.Errorf("failed: %s", err)
	}
	if len(txt) == 0 || len(txt[0]) == 0 {
		t.Errorf("no results")
	}
}

func TestGoogleDNSAddr(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}
	names, err := LookupAddr("8.8.8.8")
	if err != nil {
		t.Errorf("failed: %s", err)
	}
	if len(names) == 0 {
		t.Errorf("no results")
	}
}

func TestLookupIANACNAME(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}
	cname, err := LookupCNAME("www.iana.org")
	if !strings.HasSuffix(cname, ".icann.org.") || err != nil {
		t.Errorf(`LookupCNAME("www.iana.org.") = %q, %v, want "*.icann.org.", nil`, cname, err)
	}
}

var revAddrTests = []struct {
	Addr      string
	Reverse   string
	ErrPrefix string
}{
	{"1.2.3.4", "4.3.2.1.in-addr.arpa.", ""},
	{"245.110.36.114", "114.36.110.245.in-addr.arpa.", ""},
	{"::ffff:12.34.56.78", "78.56.34.12.in-addr.arpa.", ""},
	{"::1", "1.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.ip6.arpa.", ""},
	{"1::", "0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.1.0.0.0.ip6.arpa.", ""},
	{"1234:567::89a:bcde", "e.d.c.b.a.9.8.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.7.6.5.0.4.3.2.1.ip6.arpa.", ""},
	{"1234:567:fefe:bcbc:adad:9e4a:89a:bcde", "e.d.c.b.a.9.8.0.a.4.e.9.d.a.d.a.c.b.c.b.e.f.e.f.7.6.5.0.4.3.2.1.ip6.arpa.", ""},
	{"1.2.3", "", "unrecognized address"},
	{"1.2.3.4.5", "", "unrecognized address"},
	{"1234:567:bcbca::89a:bcde", "", "unrecognized address"},
	{"1234:567::bcbc:adad::89a:bcde", "", "unrecognized address"},
}

func TestReverseAddress(t *testing.T) {
	for i, tt := range revAddrTests {
		a, err := reverseaddr(tt.Addr)
		if len(tt.ErrPrefix) > 0 && err == nil {
			t.Errorf("#%d: expected %q, got <nil> (error)", i, tt.ErrPrefix)
			continue
		}
		if len(tt.ErrPrefix) == 0 && err != nil {
			t.Errorf("#%d: expected <nil>, got %q (error)", i, err)
		}
		if err != nil && err.(*DNSError).Err != tt.ErrPrefix {
			t.Errorf("#%d: expected %q, got %q (mismatched error)", i, tt.ErrPrefix, err.(*DNSError).Err)
		}
		if a != tt.Reverse {
			t.Errorf("#%d: expected %q, got %q (reverse address)", i, tt.Reverse, a)
		}
	}
}

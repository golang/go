// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO It would be nice to use a mock DNS server, to eliminate
// external dependencies.

package net

import (
	"flag"
	"fmt"
	"strings"
	"testing"
	"time"
)

var testExternal = flag.Bool("external", true, "allow use of external networks during long test")

var lookupGoogleSRVTests = []struct {
	service, proto, name string
	cname, target        string
}{
	{
		"xmpp-server", "tcp", "google.com",
		".google.com", ".google.com",
	},
	{
		"", "", "_xmpp-server._tcp.google.com", // non-standard back door
		".google.com", ".google.com",
	},
}

func TestLookupGoogleSRV(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}

	for _, tt := range lookupGoogleSRVTests {
		cname, srvs, err := LookupSRV(tt.service, tt.proto, tt.name)
		if err != nil {
			t.Fatal(err)
		}
		if len(srvs) == 0 {
			t.Error("got no record")
		}
		if !strings.Contains(cname, tt.cname) {
			t.Errorf("got %q; want %q", cname, tt.cname)
		}
		for _, srv := range srvs {
			if !strings.Contains(srv.Target, tt.target) {
				t.Errorf("got %v; want a record containing %q", srv, tt.target)
			}
		}
	}
}

func TestLookupGmailMX(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}

	mxs, err := LookupMX("gmail.com")
	if err != nil {
		t.Fatal(err)
	}
	if len(mxs) == 0 {
		t.Error("got no record")
	}
	for _, mx := range mxs {
		if !strings.Contains(mx.Host, ".google.com") {
			t.Errorf("got %v; want a record containing .google.com.", mx)
		}
	}
}

func TestLookupGmailNS(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}

	nss, err := LookupNS("gmail.com")
	if err != nil {
		t.Fatal(err)
	}
	if len(nss) == 0 {
		t.Error("got no record")
	}
	for _, ns := range nss {
		if !strings.Contains(ns.Host, ".google.com") {
			t.Errorf("got %v; want a record containing .google.com.", ns)
		}
	}
}

func TestLookupGmailTXT(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}

	txts, err := LookupTXT("gmail.com")
	if err != nil {
		t.Fatal(err)
	}
	if len(txts) == 0 {
		t.Error("got no record")
	}
	for _, txt := range txts {
		if !strings.Contains(txt, "spf") {
			t.Errorf("got %q; want a spf record", txt)
		}
	}
}

var lookupGooglePublicDNSAddrs = []struct {
	addr string
	name string
}{
	{"8.8.8.8", ".google.com."},
	{"8.8.4.4", ".google.com."},
	{"2001:4860:4860::8888", ".google.com."},
	{"2001:4860:4860::8844", ".google.com."},
}

func TestLookupGooglePublicDNSAddr(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}

	for _, tt := range lookupGooglePublicDNSAddrs {
		names, err := LookupAddr(tt.addr)
		if err != nil {
			t.Fatal(err)
		}
		if len(names) == 0 {
			t.Error("got no record")
		}
		for _, name := range names {
			if !strings.HasSuffix(name, tt.name) {
				t.Errorf("got %q; want a record containing %q", name, tt.name)
			}
		}
	}
}

func TestLookupIANACNAME(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}

	cname, err := LookupCNAME("www.iana.org")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(cname, ".icann.org.") {
		t.Errorf("got %q; want a record containing .icann.org.", cname)
	}
}

func TestLookupGoogleHost(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}

	addrs, err := LookupHost("google.com")
	if err != nil {
		t.Fatal(err)
	}
	if len(addrs) == 0 {
		t.Error("got no record")
	}
	for _, addr := range addrs {
		if ParseIP(addr) == nil {
			t.Errorf("got %q; want a literal ip address", addr)
		}
	}
}

func TestLookupGoogleIP(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}

	ips, err := LookupIP("google.com")
	if err != nil {
		t.Fatal(err)
	}
	if len(ips) == 0 {
		t.Error("got no record")
	}
	for _, ip := range ips {
		if ip.To4() == nil && ip.To16() == nil {
			t.Errorf("got %v; want an ip address", ip)
		}
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

var testDNSFlood = flag.Bool("dnsflood", false, "whether to test dns query flooding")

func TestLookupIPDeadline(t *testing.T) {
	if !*testDNSFlood {
		t.Skip("test disabled; use -dnsflood to enable")
	}

	const N = 5000
	const timeout = 3 * time.Second
	c := make(chan error, 2*N)
	for i := 0; i < N; i++ {
		name := fmt.Sprintf("%d.net-test.golang.org", i)
		go func() {
			_, err := lookupIPDeadline(name, time.Now().Add(timeout/2))
			c <- err
		}()
		go func() {
			_, err := lookupIPDeadline(name, time.Now().Add(timeout))
			c <- err
		}()
	}
	qstats := struct {
		succeeded, failed         int
		timeout, temporary, other int
		unknown                   int
	}{}
	deadline := time.After(timeout + time.Second)
	for i := 0; i < 2*N; i++ {
		select {
		case <-deadline:
			t.Fatal("deadline exceeded")
		case err := <-c:
			switch err := err.(type) {
			case nil:
				qstats.succeeded++
			case Error:
				qstats.failed++
				if err.Timeout() {
					qstats.timeout++
				}
				if err.Temporary() {
					qstats.temporary++
				}
				if !err.Timeout() && !err.Temporary() {
					qstats.other++
				}
			default:
				qstats.failed++
				qstats.unknown++
			}
		}
	}

	// A high volume of DNS queries for sub-domain of golang.org
	// would be coordinated by authoritative or recursive server,
	// or stub resolver which implements query-response rate
	// limitation, so we can expect some query successes and more
	// failures including timeout, temporary and other here.
	// As a rule, unknown must not be shown but it might possibly
	// happen due to issue 4856 for now.
	t.Logf("%v succeeded, %v failed (%v timeout, %v temporary, %v other, %v unknown)", qstats.succeeded, qstats.failed, qstats.timeout, qstats.temporary, qstats.other, qstats.unknown)
}

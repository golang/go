// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
	"time"
)

func lookupLocalhost(fn func(string) ([]IPAddr, error), host string) ([]IPAddr, error) {
	switch host {
	case "localhost":
		return []IPAddr{
			{IP: IPv4(127, 0, 0, 1)},
			{IP: IPv6loopback},
		}, nil
	default:
		return fn(host)
	}
}

// The Lookup APIs use various sources such as local database, DNS or
// mDNS, and may use platform-dependent DNS stub resolver if possible.
// The APIs accept any of forms for a query; host name in various
// encodings, UTF-8 encoded net name, domain name, FQDN or absolute
// FQDN, but the result would be one of the forms and it depends on
// the circumstances.

var lookupGoogleSRVTests = []struct {
	service, proto, name string
	cname, target        string
}{
	{
		"xmpp-server", "tcp", "google.com",
		"google.com", "google.com",
	},
	{
		"xmpp-server", "tcp", "google.com.",
		"google.com", "google.com",
	},

	// non-standard back door
	{
		"", "", "_xmpp-server._tcp.google.com",
		"google.com", "google.com",
	},
	{
		"", "", "_xmpp-server._tcp.google.com.",
		"google.com", "google.com",
	},
}

func TestLookupGoogleSRV(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("avoid external network")
	}
	if !supportsIPv4 || !*testIPv4 {
		t.Skip("IPv4 is required")
	}

	for _, tt := range lookupGoogleSRVTests {
		cname, srvs, err := LookupSRV(tt.service, tt.proto, tt.name)
		if err != nil {
			t.Fatal(err)
		}
		if len(srvs) == 0 {
			t.Error("got no record")
		}
		if !strings.HasSuffix(cname, tt.cname) && !strings.HasSuffix(cname, tt.cname+".") {
			t.Errorf("got %s; want %s", cname, tt.cname)
		}
		for _, srv := range srvs {
			if !strings.HasSuffix(srv.Target, tt.target) && !strings.HasSuffix(srv.Target, tt.target+".") {
				t.Errorf("got %v; want a record containing %s", srv, tt.target)
			}
		}
	}
}

var lookupGmailMXTests = []struct {
	name, host string
}{
	{"gmail.com", "google.com"},
	{"gmail.com.", "google.com"},
}

func TestLookupGmailMX(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("avoid external network")
	}
	if !supportsIPv4 || !*testIPv4 {
		t.Skip("IPv4 is required")
	}

	for _, tt := range lookupGmailMXTests {
		mxs, err := LookupMX(tt.name)
		if err != nil {
			t.Fatal(err)
		}
		if len(mxs) == 0 {
			t.Error("got no record")
		}
		for _, mx := range mxs {
			if !strings.HasSuffix(mx.Host, tt.host) && !strings.HasSuffix(mx.Host, tt.host+".") {
				t.Errorf("got %v; want a record containing %s", mx, tt.host)
			}
		}
	}
}

var lookupGmailNSTests = []struct {
	name, host string
}{
	{"gmail.com", "google.com"},
	{"gmail.com.", "google.com"},
}

func TestLookupGmailNS(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("avoid external network")
	}
	if !supportsIPv4 || !*testIPv4 {
		t.Skip("IPv4 is required")
	}

	for _, tt := range lookupGmailNSTests {
		nss, err := LookupNS(tt.name)
		if err != nil {
			t.Fatal(err)
		}
		if len(nss) == 0 {
			t.Error("got no record")
		}
		for _, ns := range nss {
			if !strings.HasSuffix(ns.Host, tt.host) && !strings.HasSuffix(ns.Host, tt.host+".") {
				t.Errorf("got %v; want a record containing %s", ns, tt.host)
			}
		}
	}
}

var lookupGmailTXTTests = []struct {
	name, txt, host string
}{
	{"gmail.com", "spf", "google.com"},
	{"gmail.com.", "spf", "google.com"},
}

func TestLookupGmailTXT(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("avoid external network")
	}
	if !supportsIPv4 || !*testIPv4 {
		t.Skip("IPv4 is required")
	}

	for _, tt := range lookupGmailTXTTests {
		txts, err := LookupTXT(tt.name)
		if err != nil {
			t.Fatal(err)
		}
		if len(txts) == 0 {
			t.Error("got no record")
		}
		for _, txt := range txts {
			if !strings.Contains(txt, tt.txt) || (!strings.HasSuffix(txt, tt.host) && !strings.HasSuffix(txt, tt.host+".")) {
				t.Errorf("got %s; want a record containing %s, %s", txt, tt.txt, tt.host)
			}
		}
	}
}

var lookupGooglePublicDNSAddrTests = []struct {
	addr, name string
}{
	{"8.8.8.8", ".google.com"},
	{"8.8.4.4", ".google.com"},
	{"2001:4860:4860::8888", ".google.com"},
	{"2001:4860:4860::8844", ".google.com"},
}

func TestLookupGooglePublicDNSAddr(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("avoid external network")
	}
	if !supportsIPv4 || !supportsIPv6 || !*testIPv4 || !*testIPv6 {
		t.Skip("both IPv4 and IPv6 are required")
	}

	for _, tt := range lookupGooglePublicDNSAddrTests {
		names, err := LookupAddr(tt.addr)
		if err != nil {
			t.Fatal(err)
		}
		if len(names) == 0 {
			t.Error("got no record")
		}
		for _, name := range names {
			if !strings.HasSuffix(name, tt.name) && !strings.HasSuffix(name, tt.name+".") {
				t.Errorf("got %s; want a record containing %s", name, tt.name)
			}
		}
	}
}

var lookupIANACNAMETests = []struct {
	name, cname string
}{
	{"www.iana.org", "icann.org"},
	{"www.iana.org.", "icann.org"},
}

func TestLookupIANACNAME(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("avoid external network")
	}
	if !supportsIPv4 || !*testIPv4 {
		t.Skip("IPv4 is required")
	}

	for _, tt := range lookupIANACNAMETests {
		cname, err := LookupCNAME(tt.name)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.HasSuffix(cname, tt.cname) && !strings.HasSuffix(cname, tt.cname+".") {
			t.Errorf("got %s; want a record containing %s", cname, tt.cname)
		}
	}
}

var lookupGoogleHostTests = []struct {
	name string
}{
	{"google.com"},
	{"google.com."},
}

func TestLookupGoogleHost(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("avoid external network")
	}
	if !supportsIPv4 || !*testIPv4 {
		t.Skip("IPv4 is required")
	}

	for _, tt := range lookupGoogleHostTests {
		addrs, err := LookupHost(tt.name)
		if err != nil {
			t.Fatal(err)
		}
		if len(addrs) == 0 {
			t.Error("got no record")
		}
		for _, addr := range addrs {
			if ParseIP(addr) == nil {
				t.Errorf("got %q; want a literal IP address", addr)
			}
		}
	}
}

var lookupGoogleIPTests = []struct {
	name string
}{
	{"google.com"},
	{"google.com."},
}

func TestLookupGoogleIP(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("avoid external network")
	}
	if !supportsIPv4 || !*testIPv4 {
		t.Skip("IPv4 is required")
	}

	for _, tt := range lookupGoogleIPTests {
		ips, err := LookupIP(tt.name)
		if err != nil {
			t.Fatal(err)
		}
		if len(ips) == 0 {
			t.Error("got no record")
		}
		for _, ip := range ips {
			if ip.To4() == nil && ip.To16() == nil {
				t.Errorf("got %v; want an IP address", ip)
			}
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

func TestLookupDots(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skipf("skipping external network test")
	}

	fixup := forceGoDNS()
	defer fixup()
	testDots(t, "go")

	if forceCgoDNS() {
		testDots(t, "cgo")
	}
}

func testDots(t *testing.T, mode string) {
	names, err := LookupAddr("8.8.8.8") // Google dns server
	if err != nil {
		t.Errorf("LookupAddr(8.8.8.8): %v (mode=%v)", err, mode)
	} else {
		for _, name := range names {
			if !strings.HasSuffix(name, ".google.com.") {
				t.Errorf("LookupAddr(8.8.8.8) = %v, want names ending in .google.com. with trailing dot (mode=%v)", names, mode)
				break
			}
		}
	}

	cname, err := LookupCNAME("www.mit.edu")
	if err != nil || !strings.HasSuffix(cname, ".") {
		t.Errorf("LookupCNAME(www.mit.edu) = %v, %v, want cname ending in . with trailing dot (mode=%v)", cname, err, mode)
	}

	mxs, err := LookupMX("google.com")
	if err != nil {
		t.Errorf("LookupMX(google.com): %v (mode=%v)", err, mode)
	} else {
		for _, mx := range mxs {
			if !strings.HasSuffix(mx.Host, ".google.com.") {
				t.Errorf("LookupMX(google.com) = %v, want names ending in .google.com. with trailing dot (mode=%v)", mxString(mxs), mode)
				break
			}
		}
	}

	nss, err := LookupNS("google.com")
	if err != nil {
		t.Errorf("LookupNS(google.com): %v (mode=%v)", err, mode)
	} else {
		for _, ns := range nss {
			if !strings.HasSuffix(ns.Host, ".google.com.") {
				t.Errorf("LookupNS(google.com) = %v, want names ending in .google.com. with trailing dot (mode=%v)", nsString(nss), mode)
				break
			}
		}
	}

	cname, srvs, err := LookupSRV("xmpp-server", "tcp", "google.com")
	if err != nil {
		t.Errorf("LookupSRV(xmpp-server, tcp, google.com): %v (mode=%v)", err, mode)
	} else {
		if !strings.HasSuffix(cname, ".google.com.") {
			t.Errorf("LookupSRV(xmpp-server, tcp, google.com) returned cname=%v, want name ending in .google.com. with trailing dot (mode=%v)", cname, mode)
		}
		for _, srv := range srvs {
			if !strings.HasSuffix(srv.Target, ".google.com.") {
				t.Errorf("LookupSRV(xmpp-server, tcp, google.com) returned addrs=%v, want names ending in .google.com. with trailing dot (mode=%v)", srvString(srvs), mode)
				break
			}
		}
	}
}

func mxString(mxs []*MX) string {
	var buf bytes.Buffer
	sep := ""
	fmt.Fprintf(&buf, "[")
	for _, mx := range mxs {
		fmt.Fprintf(&buf, "%s%s:%d", sep, mx.Host, mx.Pref)
		sep = " "
	}
	fmt.Fprintf(&buf, "]")
	return buf.String()
}

func nsString(nss []*NS) string {
	var buf bytes.Buffer
	sep := ""
	fmt.Fprintf(&buf, "[")
	for _, ns := range nss {
		fmt.Fprintf(&buf, "%s%s", sep, ns.Host)
		sep = " "
	}
	fmt.Fprintf(&buf, "]")
	return buf.String()
}

func srvString(srvs []*SRV) string {
	var buf bytes.Buffer
	sep := ""
	fmt.Fprintf(&buf, "[")
	for _, srv := range srvs {
		fmt.Fprintf(&buf, "%s%s:%d:%d:%d", sep, srv.Target, srv.Port, srv.Priority, srv.Weight)
		sep = " "
	}
	fmt.Fprintf(&buf, "]")
	return buf.String()
}

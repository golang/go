// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !js

package net

import (
	"bytes"
	"context"
	"fmt"
	"internal/testenv"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func lookupLocalhost(ctx context.Context, fn func(context.Context, string, string) ([]IPAddr, error), network, host string) ([]IPAddr, error) {
	switch host {
	case "localhost":
		return []IPAddr{
			{IP: IPv4(127, 0, 0, 1)},
			{IP: IPv6loopback},
		}, nil
	default:
		return fn(ctx, network, host)
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
		"google.com.", "google.com.",
	},
	{
		"xmpp-server", "tcp", "google.com.",
		"google.com.", "google.com.",
	},

	// non-standard back door
	{
		"", "", "_xmpp-server._tcp.google.com",
		"google.com.", "google.com.",
	},
	{
		"", "", "_xmpp-server._tcp.google.com.",
		"google.com.", "google.com.",
	},
}

var backoffDuration = [...]time.Duration{time.Second, 5 * time.Second, 30 * time.Second}

func TestLookupGoogleSRV(t *testing.T) {
	t.Parallel()
	mustHaveExternalNetwork(t)

	if runtime.GOOS == "darwin" && (runtime.GOARCH == "arm" || runtime.GOARCH == "arm64") {
		t.Skip("no resolv.conf on iOS")
	}

	if !supportsIPv4() || !*testIPv4 {
		t.Skip("IPv4 is required")
	}

	attempts := 0
	for i := 0; i < len(lookupGoogleSRVTests); i++ {
		tt := lookupGoogleSRVTests[i]
		cname, srvs, err := LookupSRV(tt.service, tt.proto, tt.name)
		if err != nil {
			testenv.SkipFlakyNet(t)
			if attempts < len(backoffDuration) {
				dur := backoffDuration[attempts]
				t.Logf("backoff %v after failure %v\n", dur, err)
				time.Sleep(dur)
				attempts++
				i--
				continue
			}
			t.Fatal(err)
		}
		if len(srvs) == 0 {
			t.Error("got no record")
		}
		if !strings.HasSuffix(cname, tt.cname) {
			t.Errorf("got %s; want %s", cname, tt.cname)
		}
		for _, srv := range srvs {
			if !strings.HasSuffix(srv.Target, tt.target) {
				t.Errorf("got %v; want a record containing %s", srv, tt.target)
			}
		}
	}
}

var lookupGmailMXTests = []struct {
	name, host string
}{
	{"gmail.com", "google.com."},
	{"gmail.com.", "google.com."},
}

func TestLookupGmailMX(t *testing.T) {
	t.Parallel()
	mustHaveExternalNetwork(t)

	if runtime.GOOS == "darwin" && (runtime.GOARCH == "arm" || runtime.GOARCH == "arm64") {
		t.Skip("no resolv.conf on iOS")
	}

	if !supportsIPv4() || !*testIPv4 {
		t.Skip("IPv4 is required")
	}

	attempts := 0
	for i := 0; i < len(lookupGmailMXTests); i++ {
		tt := lookupGmailMXTests[i]
		mxs, err := LookupMX(tt.name)
		if err != nil {
			testenv.SkipFlakyNet(t)
			if attempts < len(backoffDuration) {
				dur := backoffDuration[attempts]
				t.Logf("backoff %v after failure %v\n", dur, err)
				time.Sleep(dur)
				attempts++
				i--
				continue
			}
			t.Fatal(err)
		}
		if len(mxs) == 0 {
			t.Error("got no record")
		}
		for _, mx := range mxs {
			if !strings.HasSuffix(mx.Host, tt.host) {
				t.Errorf("got %v; want a record containing %s", mx, tt.host)
			}
		}
	}
}

var lookupGmailNSTests = []struct {
	name, host string
}{
	{"gmail.com", "google.com."},
	{"gmail.com.", "google.com."},
}

func TestLookupGmailNS(t *testing.T) {
	t.Parallel()
	mustHaveExternalNetwork(t)

	if runtime.GOOS == "darwin" && (runtime.GOARCH == "arm" || runtime.GOARCH == "arm64") {
		t.Skip("no resolv.conf on iOS")
	}

	if !supportsIPv4() || !*testIPv4 {
		t.Skip("IPv4 is required")
	}

	attempts := 0
	for i := 0; i < len(lookupGmailNSTests); i++ {
		tt := lookupGmailNSTests[i]
		nss, err := LookupNS(tt.name)
		if err != nil {
			testenv.SkipFlakyNet(t)
			if attempts < len(backoffDuration) {
				dur := backoffDuration[attempts]
				t.Logf("backoff %v after failure %v\n", dur, err)
				time.Sleep(dur)
				attempts++
				i--
				continue
			}
			t.Fatal(err)
		}
		if len(nss) == 0 {
			t.Error("got no record")
		}
		for _, ns := range nss {
			if !strings.HasSuffix(ns.Host, tt.host) {
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
	if runtime.GOOS == "plan9" {
		t.Skip("skipping on plan9; see https://golang.org/issue/29722")
	}
	t.Parallel()
	mustHaveExternalNetwork(t)

	if runtime.GOOS == "darwin" && (runtime.GOARCH == "arm" || runtime.GOARCH == "arm64") {
		t.Skip("no resolv.conf on iOS")
	}

	if !supportsIPv4() || !*testIPv4 {
		t.Skip("IPv4 is required")
	}

	attempts := 0
	for i := 0; i < len(lookupGmailTXTTests); i++ {
		tt := lookupGmailTXTTests[i]
		txts, err := LookupTXT(tt.name)
		if err != nil {
			testenv.SkipFlakyNet(t)
			if attempts < len(backoffDuration) {
				dur := backoffDuration[attempts]
				t.Logf("backoff %v after failure %v\n", dur, err)
				time.Sleep(dur)
				attempts++
				i--
				continue
			}
			t.Fatal(err)
		}
		if len(txts) == 0 {
			t.Error("got no record")
		}
		found := false
		for _, txt := range txts {
			if strings.Contains(txt, tt.txt) && (strings.HasSuffix(txt, tt.host) || strings.HasSuffix(txt, tt.host+".")) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("got %v; want a record containing %s, %s", txts, tt.txt, tt.host)
		}
	}
}

var lookupGooglePublicDNSAddrTests = []string{
	"8.8.8.8",
	"8.8.4.4",
	"2001:4860:4860::8888",
	"2001:4860:4860::8844",
}

func TestLookupGooglePublicDNSAddr(t *testing.T) {
	mustHaveExternalNetwork(t)

	if !supportsIPv4() || !supportsIPv6() || !*testIPv4 || !*testIPv6 {
		t.Skip("both IPv4 and IPv6 are required")
	}

	defer dnsWaitGroup.Wait()

	for _, ip := range lookupGooglePublicDNSAddrTests {
		names, err := LookupAddr(ip)
		if err != nil {
			t.Fatal(err)
		}
		if len(names) == 0 {
			t.Error("got no record")
		}
		for _, name := range names {
			if !strings.HasSuffix(name, ".google.com.") && !strings.HasSuffix(name, ".google.") {
				t.Errorf("got %q; want a record ending in .google.com. or .google.", name)
			}
		}
	}
}

func TestLookupIPv6LinkLocalAddr(t *testing.T) {
	if !supportsIPv6() || !*testIPv6 {
		t.Skip("IPv6 is required")
	}

	defer dnsWaitGroup.Wait()

	addrs, err := LookupHost("localhost")
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, addr := range addrs {
		if addr == "fe80::1%lo0" {
			found = true
			break
		}
	}
	if !found {
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if _, err := LookupAddr("fe80::1%lo0"); err != nil {
		t.Error(err)
	}
}

func TestLookupIPv6LinkLocalAddrWithZone(t *testing.T) {
	if !supportsIPv6() || !*testIPv6 {
		t.Skip("IPv6 is required")
	}

	ipaddrs, err := DefaultResolver.LookupIPAddr(context.Background(), "fe80::1%lo0")
	if err != nil {
		t.Error(err)
	}
	for _, addr := range ipaddrs {
		if e, a := "lo0", addr.Zone; e != a {
			t.Errorf("wrong zone: want %q, got %q", e, a)
		}
	}

	addrs, err := DefaultResolver.LookupHost(context.Background(), "fe80::1%lo0")
	if err != nil {
		t.Error(err)
	}
	for _, addr := range addrs {
		if e, a := "fe80::1%lo0", addr; e != a {
			t.Errorf("wrong host: want %q got %q", e, a)
		}
	}
}

var lookupCNAMETests = []struct {
	name, cname string
}{
	{"www.iana.org", "icann.org."},
	{"www.iana.org.", "icann.org."},
	{"www.google.com", "google.com."},
}

func TestLookupCNAME(t *testing.T) {
	mustHaveExternalNetwork(t)

	if !supportsIPv4() || !*testIPv4 {
		t.Skip("IPv4 is required")
	}

	defer dnsWaitGroup.Wait()

	attempts := 0
	for i := 0; i < len(lookupCNAMETests); i++ {
		tt := lookupCNAMETests[i]
		cname, err := LookupCNAME(tt.name)
		if err != nil {
			testenv.SkipFlakyNet(t)
			if attempts < len(backoffDuration) {
				dur := backoffDuration[attempts]
				t.Logf("backoff %v after failure %v\n", dur, err)
				time.Sleep(dur)
				attempts++
				i--
				continue
			}
			t.Fatal(err)
		}
		if !strings.HasSuffix(cname, tt.cname) {
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
	mustHaveExternalNetwork(t)

	if !supportsIPv4() || !*testIPv4 {
		t.Skip("IPv4 is required")
	}

	defer dnsWaitGroup.Wait()

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

func TestLookupLongTXT(t *testing.T) {
	testenv.SkipFlaky(t, 22857)
	mustHaveExternalNetwork(t)

	defer dnsWaitGroup.Wait()

	txts, err := LookupTXT("golang.rsc.io")
	if err != nil {
		t.Fatal(err)
	}
	sort.Strings(txts)
	want := []string{
		strings.Repeat("abcdefghijklmnopqrstuvwxyABCDEFGHJIKLMNOPQRSTUVWXY", 10),
		"gophers rule",
	}
	if !reflect.DeepEqual(txts, want) {
		t.Fatalf("LookupTXT golang.rsc.io incorrect\nhave %q\nwant %q", txts, want)
	}
}

var lookupGoogleIPTests = []struct {
	name string
}{
	{"google.com"},
	{"google.com."},
}

func TestLookupGoogleIP(t *testing.T) {
	mustHaveExternalNetwork(t)

	if !supportsIPv4() || !*testIPv4 {
		t.Skip("IPv4 is required")
	}

	defer dnsWaitGroup.Wait()

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
	defer dnsWaitGroup.Wait()
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

func TestDNSFlood(t *testing.T) {
	if !*testDNSFlood {
		t.Skip("test disabled; use -dnsflood to enable")
	}

	defer dnsWaitGroup.Wait()

	var N = 5000
	if runtime.GOOS == "darwin" {
		// On Darwin this test consumes kernel threads much
		// than other platforms for some reason.
		// When we monitor the number of allocated Ms by
		// observing on runtime.newm calls, we can see that it
		// easily reaches the per process ceiling
		// kern.num_threads when CGO_ENABLED=1 and
		// GODEBUG=netdns=go.
		N = 500
	}

	const timeout = 3 * time.Second
	ctxHalfTimeout, cancel := context.WithTimeout(context.Background(), timeout/2)
	defer cancel()
	ctxTimeout, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	c := make(chan error, 2*N)
	for i := 0; i < N; i++ {
		name := fmt.Sprintf("%d.net-test.golang.org", i)
		go func() {
			_, err := DefaultResolver.LookupIPAddr(ctxHalfTimeout, name)
			c <- err
		}()
		go func() {
			_, err := DefaultResolver.LookupIPAddr(ctxTimeout, name)
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

func TestLookupDotsWithLocalSource(t *testing.T) {
	if !supportsIPv4() || !*testIPv4 {
		t.Skip("IPv4 is required")
	}

	mustHaveExternalNetwork(t)

	defer dnsWaitGroup.Wait()

	for i, fn := range []func() func(){forceGoDNS, forceCgoDNS} {
		fixup := fn()
		if fixup == nil {
			continue
		}
		names, err := LookupAddr("127.0.0.1")
		fixup()
		if err != nil {
			t.Logf("#%d: %v", i, err)
			continue
		}
		mode := "netgo"
		if i == 1 {
			mode = "netcgo"
		}
	loop:
		for i, name := range names {
			if strings.Index(name, ".") == len(name)-1 { // "localhost" not "localhost."
				for j := range names {
					if j == i {
						continue
					}
					if names[j] == name[:len(name)-1] {
						// It's OK if we find the name without the dot,
						// as some systems say 127.0.0.1 localhost localhost.
						continue loop
					}
				}
				t.Errorf("%s: got %s; want %s", mode, name, name[:len(name)-1])
			} else if strings.Contains(name, ".") && !strings.HasSuffix(name, ".") { // "localhost.localdomain." not "localhost.localdomain"
				t.Errorf("%s: got %s; want name ending with trailing dot", mode, name)
			}
		}
	}
}

func TestLookupDotsWithRemoteSource(t *testing.T) {
	if runtime.GOOS == "darwin" {
		testenv.SkipFlaky(t, 27992)
	}
	mustHaveExternalNetwork(t)

	if !supportsIPv4() || !*testIPv4 {
		t.Skip("IPv4 is required")
	}

	if runtime.GOOS == "darwin" && (runtime.GOARCH == "arm" || runtime.GOARCH == "arm64") {
		t.Skip("no resolv.conf on iOS")
	}

	defer dnsWaitGroup.Wait()

	if fixup := forceGoDNS(); fixup != nil {
		testDots(t, "go")
		fixup()
	}
	if fixup := forceCgoDNS(); fixup != nil {
		testDots(t, "cgo")
		fixup()
	}
}

func testDots(t *testing.T, mode string) {
	names, err := LookupAddr("8.8.8.8") // Google dns server
	if err != nil {
		testenv.SkipFlakyNet(t)
		t.Errorf("LookupAddr(8.8.8.8): %v (mode=%v)", err, mode)
	} else {
		for _, name := range names {
			if !strings.HasSuffix(name, ".google.com.") && !strings.HasSuffix(name, ".google.") {
				t.Errorf("LookupAddr(8.8.8.8) = %v, want names ending in .google.com or .google with trailing dot (mode=%v)", names, mode)
				break
			}
		}
	}

	cname, err := LookupCNAME("www.mit.edu")
	if err != nil {
		testenv.SkipFlakyNet(t)
		t.Errorf("LookupCNAME(www.mit.edu, mode=%v): %v", mode, err)
	} else if !strings.HasSuffix(cname, ".") {
		t.Errorf("LookupCNAME(www.mit.edu) = %v, want cname ending in . with trailing dot (mode=%v)", cname, mode)
	}

	mxs, err := LookupMX("google.com")
	if err != nil {
		testenv.SkipFlakyNet(t)
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
		testenv.SkipFlakyNet(t)
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
		testenv.SkipFlakyNet(t)
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

func TestLookupPort(t *testing.T) {
	// See https://www.iana.org/assignments/service-names-port-numbers/service-names-port-numbers.xhtml
	//
	// Please be careful about adding new test cases.
	// There are platforms which have incomplete mappings for
	// restricted resource access and security reasons.
	type test struct {
		network string
		name    string
		port    int
		ok      bool
	}
	var tests = []test{
		{"tcp", "0", 0, true},
		{"udp", "0", 0, true},
		{"udp", "domain", 53, true},

		{"--badnet--", "zzz", 0, false},
		{"tcp", "--badport--", 0, false},
		{"tcp", "-1", 0, false},
		{"tcp", "65536", 0, false},
		{"udp", "-1", 0, false},
		{"udp", "65536", 0, false},
		{"tcp", "123456789", 0, false},

		// Issue 13610: LookupPort("tcp", "")
		{"tcp", "", 0, true},
		{"tcp4", "", 0, true},
		{"tcp6", "", 0, true},
		{"udp", "", 0, true},
		{"udp4", "", 0, true},
		{"udp6", "", 0, true},
	}

	switch runtime.GOOS {
	case "android":
		if netGo {
			t.Skipf("not supported on %s without cgo; see golang.org/issues/14576", runtime.GOOS)
		}
	default:
		tests = append(tests, test{"tcp", "http", 80, true})
	}

	for _, tt := range tests {
		port, err := LookupPort(tt.network, tt.name)
		if port != tt.port || (err == nil) != tt.ok {
			t.Errorf("LookupPort(%q, %q) = %d, %v; want %d, error=%t", tt.network, tt.name, port, err, tt.port, !tt.ok)
		}
		if err != nil {
			if perr := parseLookupPortError(err); perr != nil {
				t.Error(perr)
			}
		}
	}
}

// Like TestLookupPort but with minimal tests that should always pass
// because the answers are baked-in to the net package.
func TestLookupPort_Minimal(t *testing.T) {
	type test struct {
		network string
		name    string
		port    int
	}
	var tests = []test{
		{"tcp", "http", 80},
		{"tcp", "HTTP", 80}, // case shouldn't matter
		{"tcp", "https", 443},
		{"tcp", "ssh", 22},
		{"tcp", "gopher", 70},
		{"tcp4", "http", 80},
		{"tcp6", "http", 80},
	}

	for _, tt := range tests {
		port, err := LookupPort(tt.network, tt.name)
		if port != tt.port || err != nil {
			t.Errorf("LookupPort(%q, %q) = %d, %v; want %d, error=nil", tt.network, tt.name, port, err, tt.port)
		}
	}
}

func TestLookupProtocol_Minimal(t *testing.T) {
	type test struct {
		name string
		want int
	}
	var tests = []test{
		{"tcp", 6},
		{"TcP", 6}, // case shouldn't matter
		{"icmp", 1},
		{"igmp", 2},
		{"udp", 17},
		{"ipv6-icmp", 58},
	}

	for _, tt := range tests {
		got, err := lookupProtocol(context.Background(), tt.name)
		if got != tt.want || err != nil {
			t.Errorf("LookupProtocol(%q) = %d, %v; want %d, error=nil", tt.name, got, err, tt.want)
		}
	}

}

func TestLookupNonLDH(t *testing.T) {
	if runtime.GOOS == "nacl" {
		t.Skip("skip on nacl")
	}

	defer dnsWaitGroup.Wait()

	if fixup := forceGoDNS(); fixup != nil {
		defer fixup()
	}

	// "LDH" stands for letters, digits, and hyphens and is the usual
	// description of standard DNS names.
	// This test is checking that other kinds of names are reported
	// as not found, not reported as invalid names.
	addrs, err := LookupHost("!!!.###.bogus..domain.")
	if err == nil {
		t.Fatalf("lookup succeeded: %v", addrs)
	}
	if !strings.HasSuffix(err.Error(), errNoSuchHost.Error()) {
		t.Fatalf("lookup error = %v, want %v", err, errNoSuchHost)
	}
}

func TestLookupContextCancel(t *testing.T) {
	mustHaveExternalNetwork(t)
	if runtime.GOOS == "nacl" {
		t.Skip("skip on nacl")
	}

	defer dnsWaitGroup.Wait()

	ctx, ctxCancel := context.WithCancel(context.Background())
	ctxCancel()
	_, err := DefaultResolver.LookupIPAddr(ctx, "google.com")
	if err != errCanceled {
		testenv.SkipFlakyNet(t)
		t.Fatal(err)
	}
	ctx = context.Background()
	_, err = DefaultResolver.LookupIPAddr(ctx, "google.com")
	if err != nil {
		testenv.SkipFlakyNet(t)
		t.Fatal(err)
	}
}

// Issue 24330: treat the nil *Resolver like a zero value. Verify nothing
// crashes if nil is used.
func TestNilResolverLookup(t *testing.T) {
	mustHaveExternalNetwork(t)
	if runtime.GOOS == "nacl" {
		t.Skip("skip on nacl")
	}
	var r *Resolver = nil
	ctx := context.Background()

	// Don't care about the results, just that nothing panics:
	r.LookupAddr(ctx, "8.8.8.8")
	r.LookupCNAME(ctx, "google.com")
	r.LookupHost(ctx, "google.com")
	r.LookupIPAddr(ctx, "google.com")
	r.LookupMX(ctx, "gmail.com")
	r.LookupNS(ctx, "google.com")
	r.LookupPort(ctx, "tcp", "smtp")
	r.LookupSRV(ctx, "service", "proto", "name")
	r.LookupTXT(ctx, "gmail.com")
}

// TestLookupHostCancel verifies that lookup works even after many
// canceled lookups (see golang.org/issue/24178 for details).
func TestLookupHostCancel(t *testing.T) {
	mustHaveExternalNetwork(t)
	if runtime.GOOS == "nacl" {
		t.Skip("skip on nacl")
	}

	const (
		google        = "www.google.com"
		invalidDomain = "invalid.invalid" // RFC 2606 reserves .invalid
		n             = 600               // this needs to be larger than threadLimit size
	)

	_, err := LookupHost(google)
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	for i := 0; i < n; i++ {
		addr, err := DefaultResolver.LookupHost(ctx, invalidDomain)
		if err == nil {
			t.Fatalf("LookupHost(%q): returns %v, but should fail", invalidDomain, addr)
		}
		if !strings.Contains(err.Error(), "canceled") {
			t.Fatalf("LookupHost(%q): failed with unexpected error: %v", invalidDomain, err)
		}
		time.Sleep(time.Millisecond * 1)
	}

	_, err = LookupHost(google)
	if err != nil {
		t.Fatal(err)
	}
}

type lookupCustomResolver struct {
	*Resolver
	mu     sync.RWMutex
	dialed bool
}

func (lcr *lookupCustomResolver) dial() func(ctx context.Context, network, address string) (Conn, error) {
	return func(ctx context.Context, network, address string) (Conn, error) {
		lcr.mu.Lock()
		lcr.dialed = true
		lcr.mu.Unlock()
		return Dial(network, address)
	}
}

// TestConcurrentPreferGoResolversDial tests that multiple resolvers with the
// PreferGo option used concurrently are all dialed properly.
func TestConcurrentPreferGoResolversDial(t *testing.T) {
	// The windows implementation of the resolver does not use the Dial
	// function.
	if runtime.GOOS == "windows" {
		t.Skip("skip on windows")
	}

	testenv.MustHaveExternalNetwork(t)
	testenv.SkipFlakyNet(t)

	defer dnsWaitGroup.Wait()

	resolvers := make([]*lookupCustomResolver, 2)
	for i := range resolvers {
		cs := lookupCustomResolver{Resolver: &Resolver{PreferGo: true}}
		cs.Dial = cs.dial()
		resolvers[i] = &cs
	}

	var wg sync.WaitGroup
	wg.Add(len(resolvers))
	for i, resolver := range resolvers {
		go func(r *Resolver, index int) {
			defer wg.Done()
			_, err := r.LookupIPAddr(context.Background(), "google.com")
			if err != nil {
				t.Fatalf("lookup failed for resolver %d: %q", index, err)
			}
		}(resolver.Resolver, i)
	}
	wg.Wait()

	for i, resolver := range resolvers {
		if !resolver.dialed {
			t.Errorf("custom resolver %d not dialed during lookup", i)
		}
	}
}

var ipVersionTests = []struct {
	network string
	version byte
}{
	{"tcp", 0},
	{"tcp4", '4'},
	{"tcp6", '6'},
	{"udp", 0},
	{"udp4", '4'},
	{"udp6", '6'},
	{"ip", 0},
	{"ip4", '4'},
	{"ip6", '6'},
	{"ip7", 0},
	{"", 0},
}

func TestIPVersion(t *testing.T) {
	for _, tt := range ipVersionTests {
		if version := ipVersion(tt.network); version != tt.version {
			t.Errorf("Family for: %s. Expected: %s, Got: %s", tt.network,
				string(tt.version), string(version))
		}
	}
}

// Issue 28600: The context that is used to lookup ips should always
// preserve the values from the context that was passed into LookupIPAddr.
func TestLookupIPAddrPreservesContextValues(t *testing.T) {
	origTestHookLookupIP := testHookLookupIP
	defer func() { testHookLookupIP = origTestHookLookupIP }()

	keyValues := []struct {
		key, value interface{}
	}{
		{"key-1", 12},
		{384, "value2"},
		{new(float64), 137},
	}
	ctx := context.Background()
	for _, kv := range keyValues {
		ctx = context.WithValue(ctx, kv.key, kv.value)
	}

	wantIPs := []IPAddr{
		{IP: IPv4(127, 0, 0, 1)},
		{IP: IPv6loopback},
	}

	checkCtxValues := func(ctx_ context.Context, fn func(context.Context, string, string) ([]IPAddr, error), network, host string) ([]IPAddr, error) {
		for _, kv := range keyValues {
			g, w := ctx_.Value(kv.key), kv.value
			if !reflect.DeepEqual(g, w) {
				t.Errorf("Value lookup:\n\tGot:  %v\n\tWant: %v", g, w)
			}
		}
		return wantIPs, nil
	}
	testHookLookupIP = checkCtxValues

	resolvers := []*Resolver{
		nil,
		new(Resolver),
	}

	for i, resolver := range resolvers {
		gotIPs, err := resolver.LookupIPAddr(ctx, "golang.org")
		if err != nil {
			t.Errorf("Resolver #%d: unexpected error: %v", i, err)
		}
		if !reflect.DeepEqual(gotIPs, wantIPs) {
			t.Errorf("#%d: mismatched IPAddr results\n\tGot: %v\n\tWant: %v", i, gotIPs, wantIPs)
		}
	}
}

// Issue 30521: The lookup group should call the resolver for each network.
func TestLookupIPAddrConcurrentCallsForNetworks(t *testing.T) {
	origTestHookLookupIP := testHookLookupIP
	defer func() { testHookLookupIP = origTestHookLookupIP }()

	queries := [][]string{
		{"udp", "golang.org"},
		{"udp4", "golang.org"},
		{"udp6", "golang.org"},
		{"udp", "golang.org"},
		{"udp", "golang.org"},
	}
	results := map[[2]string][]IPAddr{
		{"udp", "golang.org"}: {
			{IP: IPv4(127, 0, 0, 1)},
			{IP: IPv6loopback},
		},
		{"udp4", "golang.org"}: {
			{IP: IPv4(127, 0, 0, 1)},
		},
		{"udp6", "golang.org"}: {
			{IP: IPv6loopback},
		},
	}
	calls := int32(0)
	waitCh := make(chan struct{})
	testHookLookupIP = func(ctx context.Context, fn func(context.Context, string, string) ([]IPAddr, error), network, host string) ([]IPAddr, error) {
		// We'll block until this is called one time for each different
		// expected result. This will ensure that the lookup group would wait
		// for the existing call if it was to be reused.
		if atomic.AddInt32(&calls, 1) == int32(len(results)) {
			close(waitCh)
		}
		select {
		case <-waitCh:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		return results[[2]string{network, host}], nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	wg := sync.WaitGroup{}
	for _, q := range queries {
		network := q[0]
		host := q[1]
		wg.Add(1)
		go func() {
			defer wg.Done()
			gotIPs, err := DefaultResolver.lookupIPAddr(ctx, network, host)
			if err != nil {
				t.Errorf("lookupIPAddr(%v, %v): unexpected error: %v", network, host, err)
			}
			wantIPs := results[[2]string{network, host}]
			if !reflect.DeepEqual(gotIPs, wantIPs) {
				t.Errorf("lookupIPAddr(%v, %v): mismatched IPAddr results\n\tGot: %v\n\tWant: %v", network, host, gotIPs, wantIPs)
			}
		}()
	}
	wg.Wait()
}

func TestWithUnexpiredValuesPreserved(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	// Insert a value into it.
	key, value := "key-1", 2
	ctx = context.WithValue(ctx, key, value)

	// Now use the "values preserving context" like
	// we would for LookupIPAddr. See Issue 28600.
	ctx = withUnexpiredValuesPreserved(ctx)

	// Lookup before expiry.
	if g, w := ctx.Value(key), value; g != w {
		t.Errorf("Lookup before expiry: Got %v Want %v", g, w)
	}

	// Cancel the context.
	cancel()

	// Lookup after expiry should return nil
	if g := ctx.Value(key); g != nil {
		t.Errorf("Lookup after expiry: Got %v want nil", g)
	}
}

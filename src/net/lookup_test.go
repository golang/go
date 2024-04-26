// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"context"
	"errors"
	"fmt"
	"internal/testenv"
	"net/netip"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

var goResolver = Resolver{PreferGo: true}

func hasSuffixFold(s, suffix string) bool {
	return strings.HasSuffix(strings.ToLower(s), strings.ToLower(suffix))
}

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
		"ldap", "tcp", "google.com",
		"google.com.", "google.com.",
	},
	{
		"ldap", "tcp", "google.com.",
		"google.com.", "google.com.",
	},

	// non-standard back door
	{
		"", "", "_ldap._tcp.google.com",
		"google.com.", "google.com.",
	},
	{
		"", "", "_ldap._tcp.google.com.",
		"google.com.", "google.com.",
	},
}

var backoffDuration = [...]time.Duration{time.Second, 5 * time.Second, 30 * time.Second}

func TestLookupGoogleSRV(t *testing.T) {
	t.Parallel()
	mustHaveExternalNetwork(t)

	if runtime.GOOS == "ios" {
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
		if !hasSuffixFold(cname, tt.cname) {
			t.Errorf("got %s; want %s", cname, tt.cname)
		}
		for _, srv := range srvs {
			if !hasSuffixFold(srv.Target, tt.target) {
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

	if runtime.GOOS == "ios" {
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
			if !hasSuffixFold(mx.Host, tt.host) {
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

	if runtime.GOOS == "ios" {
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
			if !hasSuffixFold(ns.Host, tt.host) {
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

	if runtime.GOOS == "ios" {
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
			if !hasSuffixFold(name, ".google.com.") && !hasSuffixFold(name, ".google.") {
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
	{"google.com", "google.com."},
	{"cname-to-txt.go4.org", "test-txt-record.go4.org."},
}

func TestLookupCNAME(t *testing.T) {
	mustHaveExternalNetwork(t)
	testenv.SkipFlakyNet(t)

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
		if !hasSuffixFold(cname, tt.cname) {
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
	testenv.SkipFlakyNet(t)

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
	testenv.SkipFlakyNet(t)

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
	if runtime.GOOS == "darwin" || runtime.GOOS == "ios" {
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
	if runtime.GOOS == "darwin" || runtime.GOOS == "ios" {
		testenv.SkipFlaky(t, 27992)
	}
	mustHaveExternalNetwork(t)
	testenv.SkipFlakyNet(t)

	if !supportsIPv4() || !*testIPv4 {
		t.Skip("IPv4 is required")
	}

	if runtime.GOOS == "ios" {
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
		t.Errorf("LookupAddr(8.8.8.8): %v (mode=%v)", err, mode)
	} else {
		for _, name := range names {
			if !hasSuffixFold(name, ".google.com.") && !hasSuffixFold(name, ".google.") {
				t.Errorf("LookupAddr(8.8.8.8) = %v, want names ending in .google.com or .google with trailing dot (mode=%v)", names, mode)
				break
			}
		}
	}

	cname, err := LookupCNAME("www.mit.edu")
	if err != nil {
		t.Errorf("LookupCNAME(www.mit.edu, mode=%v): %v", mode, err)
	} else if !strings.HasSuffix(cname, ".") {
		t.Errorf("LookupCNAME(www.mit.edu) = %v, want cname ending in . with trailing dot (mode=%v)", cname, mode)
	}

	mxs, err := LookupMX("google.com")
	if err != nil {
		t.Errorf("LookupMX(google.com): %v (mode=%v)", err, mode)
	} else {
		for _, mx := range mxs {
			if !hasSuffixFold(mx.Host, ".google.com.") {
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
			if !hasSuffixFold(ns.Host, ".google.com.") {
				t.Errorf("LookupNS(google.com) = %v, want names ending in .google.com. with trailing dot (mode=%v)", nsString(nss), mode)
				break
			}
		}
	}

	cname, srvs, err := LookupSRV("ldap", "tcp", "google.com")
	if err != nil {
		t.Errorf("LookupSRV(ldap, tcp, google.com): %v (mode=%v)", err, mode)
	} else {
		if !hasSuffixFold(cname, ".google.com.") {
			t.Errorf("LookupSRV(ldap, tcp, google.com) returned cname=%v, want name ending in .google.com. with trailing dot (mode=%v)", cname, mode)
		}
		for _, srv := range srvs {
			if !hasSuffixFold(srv.Target, ".google.com.") {
				t.Errorf("LookupSRV(ldap, tcp, google.com) returned addrs=%v, want names ending in .google.com. with trailing dot (mode=%v)", srvString(srvs), mode)
				break
			}
		}
	}
}

func mxString(mxs []*MX) string {
	var buf strings.Builder
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
	var buf strings.Builder
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
	var buf strings.Builder
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
		if netGoBuildTag {
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
	if !err.(*DNSError).IsNotFound {
		t.Fatalf("lookup error = %v, want true", err.(*DNSError).IsNotFound)
	}
}

func TestLookupContextCancel(t *testing.T) {
	mustHaveExternalNetwork(t)
	testenv.SkipFlakyNet(t)

	origTestHookLookupIP := testHookLookupIP
	defer func() {
		dnsWaitGroup.Wait()
		testHookLookupIP = origTestHookLookupIP
	}()

	lookupCtx, cancelLookup := context.WithCancel(context.Background())
	unblockLookup := make(chan struct{})

	// Set testHookLookupIP to start a new, concurrent call to LookupIPAddr
	// and cancel the original one, then block until the canceled call has returned
	// (ensuring that it has performed any synchronous cleanup).
	testHookLookupIP = func(
		ctx context.Context,
		fn func(context.Context, string, string) ([]IPAddr, error),
		network string,
		host string,
	) ([]IPAddr, error) {
		select {
		case <-unblockLookup:
		default:
			// Start a concurrent LookupIPAddr for the same host while the caller is
			// still blocked, and sleep a little to give it time to be deduplicated
			// before we cancel (and unblock) the caller.
			// (If the timing doesn't quite work out, we'll end up testing sequential
			// calls instead of concurrent ones, but the test should still pass.)
			t.Logf("starting concurrent LookupIPAddr")
			dnsWaitGroup.Add(1)
			go func() {
				defer dnsWaitGroup.Done()
				_, err := DefaultResolver.LookupIPAddr(context.Background(), host)
				if err != nil {
					t.Error(err)
				}
			}()
			time.Sleep(1 * time.Millisecond)
		}

		cancelLookup()
		<-unblockLookup
		// If the concurrent lookup above is deduplicated to this one
		// (as we expect to happen most of the time), it is important
		// that the original call does not cancel the shared Context.
		// (See https://go.dev/issue/22724.) Explicitly check for
		// cancellation now, just in case fn itself doesn't notice it.
		if err := ctx.Err(); err != nil {
			t.Logf("testHookLookupIP canceled")
			return nil, err
		}
		t.Logf("testHookLookupIP performing lookup")
		return fn(ctx, network, host)
	}

	_, err := DefaultResolver.LookupIPAddr(lookupCtx, "google.com")
	if dnsErr, ok := err.(*DNSError); !ok || dnsErr.Err != errCanceled.Error() {
		t.Errorf("unexpected error from canceled, blocked LookupIPAddr: %v", err)
	}
	close(unblockLookup)
}

// Issue 24330: treat the nil *Resolver like a zero value. Verify nothing
// crashes if nil is used.
func TestNilResolverLookup(t *testing.T) {
	mustHaveExternalNetwork(t)
	var r *Resolver = nil
	ctx := context.Background()

	// Don't care about the results, just that nothing panics:
	r.LookupAddr(ctx, "8.8.8.8")
	r.LookupCNAME(ctx, "google.com")
	r.LookupHost(ctx, "google.com")
	r.LookupIPAddr(ctx, "google.com")
	r.LookupIP(ctx, "ip", "google.com")
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
	testenv.SkipFlakyNet(t)
	t.Parallel() // Executes 600ms worth of sequential sleeps.

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

		// Don't verify what the actual error is.
		// We know that it must be non-nil because the domain is invalid,
		// but we don't have any guarantee that LookupHost actually bothers
		// to check for cancellation on the fast path.
		// (For example, it could use a local cache to avoid blocking entirely.)

		// The lookup may deduplicate in-flight requests, so give it time to settle
		// in between.
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
	switch runtime.GOOS {
	case "plan9":
		// TODO: plan9 implementation of the resolver uses the Dial function since
		// https://go.dev/cl/409234, this test could probably be reenabled.
		t.Skipf("skip on %v", runtime.GOOS)
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
				t.Errorf("lookup failed for resolver %d: %q", index, err)
			}
		}(resolver.Resolver, i)
	}
	wg.Wait()

	if t.Failed() {
		t.FailNow()
	}

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
		key, value any
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

// Issue 53995: Resolver.LookupIP should return error for empty host name.
func TestResolverLookupIPWithEmptyHost(t *testing.T) {
	_, err := DefaultResolver.LookupIP(context.Background(), "ip", "")
	if err == nil {
		t.Fatal("DefaultResolver.LookupIP for empty host success, want no host error")
	}
	if !strings.HasSuffix(err.Error(), errNoSuchHost.Error()) {
		t.Fatalf("lookup error = %v, want %v", err, errNoSuchHost)
	}
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

// Issue 31597: don't panic on null byte in name
func TestLookupNullByte(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)
	testenv.SkipFlakyNet(t)
	LookupHost("foo\x00bar") // check that it doesn't panic; it used to on Windows
}

func TestResolverLookupIP(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	v4Ok := supportsIPv4() && *testIPv4
	v6Ok := supportsIPv6() && *testIPv6

	defer dnsWaitGroup.Wait()

	for _, impl := range []struct {
		name string
		fn   func() func()
	}{
		{"go", forceGoDNS},
		{"cgo", forceCgoDNS},
	} {
		t.Run("implementation: "+impl.name, func(t *testing.T) {
			fixup := impl.fn()
			if fixup == nil {
				t.Skip("not supported")
			}
			defer fixup()

			for _, network := range []string{"ip", "ip4", "ip6"} {
				t.Run("network: "+network, func(t *testing.T) {
					switch {
					case network == "ip4" && !v4Ok:
						t.Skip("IPv4 is not supported")
					case network == "ip6" && !v6Ok:
						t.Skip("IPv6 is not supported")
					}

					// google.com has both A and AAAA records.
					const host = "google.com"
					ips, err := DefaultResolver.LookupIP(context.Background(), network, host)
					if err != nil {
						testenv.SkipFlakyNet(t)
						t.Fatalf("DefaultResolver.LookupIP(%q, %q): failed with unexpected error: %v", network, host, err)
					}

					var v4Addrs []netip.Addr
					var v6Addrs []netip.Addr
					for _, ip := range ips {
						if addr, ok := netip.AddrFromSlice(ip); ok {
							if addr.Is4() {
								v4Addrs = append(v4Addrs, addr)
							} else {
								v6Addrs = append(v6Addrs, addr)
							}
						} else {
							t.Fatalf("IP=%q is neither IPv4 nor IPv6", ip)
						}
					}

					// Check that we got the expected addresses.
					if network == "ip4" || network == "ip" && v4Ok {
						if len(v4Addrs) == 0 {
							t.Errorf("DefaultResolver.LookupIP(%q, %q): no IPv4 addresses", network, host)
						}
					}
					if network == "ip6" || network == "ip" && v6Ok {
						if len(v6Addrs) == 0 {
							t.Errorf("DefaultResolver.LookupIP(%q, %q): no IPv6 addresses", network, host)
						}
					}

					// Check that we didn't get any unexpected addresses.
					if network == "ip6" && len(v4Addrs) > 0 {
						t.Errorf("DefaultResolver.LookupIP(%q, %q): unexpected IPv4 addresses: %v", network, host, v4Addrs)
					}
					if network == "ip4" && len(v6Addrs) > 0 {
						t.Errorf("DefaultResolver.LookupIP(%q, %q): unexpected IPv6 or IPv4-mapped IPv6 addresses: %v", network, host, v6Addrs)
					}
				})
			}
		})
	}
}

// A context timeout should still return a DNSError.
func TestDNSTimeout(t *testing.T) {
	origTestHookLookupIP := testHookLookupIP
	defer func() { testHookLookupIP = origTestHookLookupIP }()
	defer dnsWaitGroup.Wait()

	timeoutHookGo := make(chan bool, 1)
	timeoutHook := func(ctx context.Context, fn func(context.Context, string, string) ([]IPAddr, error), network, host string) ([]IPAddr, error) {
		<-timeoutHookGo
		return nil, context.DeadlineExceeded
	}
	testHookLookupIP = timeoutHook

	checkErr := func(err error) {
		t.Helper()
		if err == nil {
			t.Error("expected an error")
		} else if dnserr, ok := err.(*DNSError); !ok {
			t.Errorf("got error type %T, want %T", err, (*DNSError)(nil))
		} else if !dnserr.IsTimeout {
			t.Errorf("got error %#v, want IsTimeout == true", dnserr)
		} else if isTimeout := dnserr.Timeout(); !isTimeout {
			t.Errorf("got err.Timeout() == %t, want true", isTimeout)
		}
	}

	// Single lookup.
	timeoutHookGo <- true
	_, err := LookupIP("golang.org")
	checkErr(err)

	// Double lookup.
	var err1, err2 error
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		_, err1 = LookupIP("golang1.org")
	}()
	go func() {
		defer wg.Done()
		_, err2 = LookupIP("golang1.org")
	}()
	close(timeoutHookGo)
	wg.Wait()
	checkErr(err1)
	checkErr(err2)

	// Double lookup with context.
	timeoutHookGo = make(chan bool)
	ctx, cancel := context.WithTimeout(context.Background(), time.Nanosecond)
	wg.Add(2)
	go func() {
		defer wg.Done()
		_, err1 = DefaultResolver.LookupIPAddr(ctx, "golang2.org")
	}()
	go func() {
		defer wg.Done()
		_, err2 = DefaultResolver.LookupIPAddr(ctx, "golang2.org")
	}()
	time.Sleep(10 * time.Nanosecond)
	close(timeoutHookGo)
	wg.Wait()
	checkErr(err1)
	checkErr(err2)
	cancel()
}

func TestLookupNoData(t *testing.T) {
	if runtime.GOOS == "plan9" {
		t.Skip("not supported on plan9")
	}

	mustHaveExternalNetwork(t)

	testLookupNoData(t, "default resolver")

	func() {
		defer forceGoDNS()()
		testLookupNoData(t, "forced go resolver")
	}()

	func() {
		defer forceCgoDNS()()
		testLookupNoData(t, "forced cgo resolver")
	}()
}

func testLookupNoData(t *testing.T, prefix string) {
	attempts := 0
	for {
		// Domain that doesn't have any A/AAAA RRs, but has different one (in this case a TXT),
		// so that it returns an empty response without any error codes (NXDOMAIN).
		_, err := LookupHost("golang.rsc.io.")
		if err == nil {
			t.Errorf("%v: unexpected success", prefix)
			return
		}

		var dnsErr *DNSError
		if errors.As(err, &dnsErr) {
			succeeded := true
			if !dnsErr.IsNotFound {
				succeeded = false
				t.Logf("%v: IsNotFound is set to false", prefix)
			}

			if dnsErr.Err != errNoSuchHost.Error() {
				succeeded = false
				t.Logf("%v: error message is not equal to: %v", prefix, errNoSuchHost.Error())
			}

			if succeeded {
				return
			}
		}

		testenv.SkipFlakyNet(t)
		if attempts < len(backoffDuration) {
			dur := backoffDuration[attempts]
			t.Logf("%v: backoff %v after failure %v\n", prefix, dur, err)
			time.Sleep(dur)
			attempts++
			continue
		}

		t.Errorf("%v: unexpected error: %v", prefix, err)
		return
	}
}

func TestLookupPortNotFound(t *testing.T) {
	allResolvers(t, func(t *testing.T) {
		_, err := LookupPort("udp", "_-unknown-service-")
		var dnsErr *DNSError
		if !errors.As(err, &dnsErr) || !dnsErr.IsNotFound {
			t.Fatalf("unexpected error: %v", err)
		}
	})
}

// submissions service is only available through a tcp network, see:
// https://www.iana.org/assignments/service-names-port-numbers/service-names-port-numbers.xhtml?search=submissions
var tcpOnlyService = func() string {
	// plan9 does not have submissions service defined in the service database.
	if runtime.GOOS == "plan9" {
		return "https"
	}
	return "submissions"
}()

func TestLookupPortDifferentNetwork(t *testing.T) {
	allResolvers(t, func(t *testing.T) {
		_, err := LookupPort("udp", tcpOnlyService)
		var dnsErr *DNSError
		if !errors.As(err, &dnsErr) || !dnsErr.IsNotFound {
			t.Fatalf("unexpected error: %v", err)
		}
	})
}

func TestLookupPortEmptyNetworkString(t *testing.T) {
	allResolvers(t, func(t *testing.T) {
		_, err := LookupPort("", tcpOnlyService)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})
}

func TestLookupPortIPNetworkString(t *testing.T) {
	allResolvers(t, func(t *testing.T) {
		_, err := LookupPort("ip", tcpOnlyService)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})
}

func TestLookupNoSuchHost(t *testing.T) {
	mustHaveExternalNetwork(t)

	const testNXDOMAIN = "invalid.invalid."
	const testNODATA = "_ldap._tcp.google.com."

	tests := []struct {
		name  string
		query func() error
	}{
		{
			name: "LookupCNAME NXDOMAIN",
			query: func() error {
				_, err := LookupCNAME(testNXDOMAIN)
				return err
			},
		},
		{
			name: "LookupHost NXDOMAIN",
			query: func() error {
				_, err := LookupHost(testNXDOMAIN)
				return err
			},
		},
		{
			name: "LookupHost NODATA",
			query: func() error {
				_, err := LookupHost(testNODATA)
				return err
			},
		},
		{
			name: "LookupMX NXDOMAIN",
			query: func() error {
				_, err := LookupMX(testNXDOMAIN)
				return err
			},
		},
		{
			name: "LookupMX NODATA",
			query: func() error {
				_, err := LookupMX(testNODATA)
				return err
			},
		},
		{
			name: "LookupNS NXDOMAIN",
			query: func() error {
				_, err := LookupNS(testNXDOMAIN)
				return err
			},
		},
		{
			name: "LookupNS NODATA",
			query: func() error {
				_, err := LookupNS(testNODATA)
				return err
			},
		},
		{
			name: "LookupSRV NXDOMAIN",
			query: func() error {
				_, _, err := LookupSRV("unknown", "tcp", testNXDOMAIN)
				return err
			},
		},
		{
			name: "LookupTXT NXDOMAIN",
			query: func() error {
				_, err := LookupTXT(testNXDOMAIN)
				return err
			},
		},
		{
			name: "LookupTXT NODATA",
			query: func() error {
				_, err := LookupTXT(testNODATA)
				return err
			},
		},
	}

	for _, v := range tests {
		t.Run(v.name, func(t *testing.T) {
			allResolvers(t, func(t *testing.T) {
				attempts := 0
				for {
					err := v.query()
					if err == nil {
						t.Errorf("unexpected success")
						return
					}
					if dnsErr, ok := err.(*DNSError); ok {
						succeeded := true
						if !dnsErr.IsNotFound {
							succeeded = false
							t.Log("IsNotFound is set to false")
						}
						if dnsErr.Err != errNoSuchHost.Error() {
							succeeded = false
							t.Logf("error message is not equal to: %v", errNoSuchHost.Error())
						}
						if succeeded {
							return
						}
					}
					testenv.SkipFlakyNet(t)
					if attempts < len(backoffDuration) {
						dur := backoffDuration[attempts]
						t.Logf("backoff %v after failure %v\n", dur, err)
						time.Sleep(dur)
						attempts++
						continue
					}
					t.Errorf("unexpected error: %v", err)
					return
				}
			})
		})
	}
}

func TestDNSErrorUnwrap(t *testing.T) {
	rDeadlineExcceeded := &Resolver{PreferGo: true, Dial: func(ctx context.Context, network, address string) (Conn, error) {
		return nil, context.DeadlineExceeded
	}}
	rCancelled := &Resolver{PreferGo: true, Dial: func(ctx context.Context, network, address string) (Conn, error) {
		return nil, context.Canceled
	}}

	_, err := rDeadlineExcceeded.LookupHost(context.Background(), "test.go.dev")
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("errors.Is(err, context.DeadlineExceeded) = false; want = true")
	}

	_, err = rCancelled.LookupHost(context.Background(), "test.go.dev")
	if !errors.Is(err, context.Canceled) {
		t.Errorf("errors.Is(err, context.Canceled) = false; want = true")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = goResolver.LookupHost(ctx, "text.go.dev")
	if !errors.Is(err, context.Canceled) {
		t.Errorf("errors.Is(err, context.Canceled) = false; want = true")
	}
}

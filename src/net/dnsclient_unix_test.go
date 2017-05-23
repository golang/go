// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package net

import (
	"context"
	"fmt"
	"internal/testenv"
	"io/ioutil"
	"os"
	"path"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

// Test address from 192.0.2.0/24 block, reserved by RFC 5737 for documentation.
const TestAddr uint32 = 0xc0000201

var dnsTransportFallbackTests = []struct {
	server  string
	name    string
	qtype   uint16
	timeout int
	rcode   int
}{
	// Querying "com." with qtype=255 usually makes an answer
	// which requires more than 512 bytes.
	{"8.8.8.8:53", "com.", dnsTypeALL, 2, dnsRcodeSuccess},
	{"8.8.4.4:53", "com.", dnsTypeALL, 4, dnsRcodeSuccess},
}

func TestDNSTransportFallback(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	for _, tt := range dnsTransportFallbackTests {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		msg, err := exchange(ctx, tt.server, tt.name, tt.qtype, time.Second)
		if err != nil {
			t.Error(err)
			continue
		}
		switch msg.rcode {
		case tt.rcode, dnsRcodeServerFailure:
		default:
			t.Errorf("got %v from %v; want %v", msg.rcode, tt.server, tt.rcode)
			continue
		}
	}
}

// See RFC 6761 for further information about the reserved, pseudo
// domain names.
var specialDomainNameTests = []struct {
	name  string
	qtype uint16
	rcode int
}{
	// Name resolution APIs and libraries should not recognize the
	// followings as special.
	{"1.0.168.192.in-addr.arpa.", dnsTypePTR, dnsRcodeNameError},
	{"test.", dnsTypeALL, dnsRcodeNameError},
	{"example.com.", dnsTypeALL, dnsRcodeSuccess},

	// Name resolution APIs and libraries should recognize the
	// followings as special and should not send any queries.
	// Though, we test those names here for verifying negative
	// answers at DNS query-response interaction level.
	{"localhost.", dnsTypeALL, dnsRcodeNameError},
	{"invalid.", dnsTypeALL, dnsRcodeNameError},
}

func TestSpecialDomainName(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	server := "8.8.8.8:53"
	for _, tt := range specialDomainNameTests {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		msg, err := exchange(ctx, server, tt.name, tt.qtype, 3*time.Second)
		if err != nil {
			t.Error(err)
			continue
		}
		switch msg.rcode {
		case tt.rcode, dnsRcodeServerFailure:
		default:
			t.Errorf("got %v from %v; want %v", msg.rcode, server, tt.rcode)
			continue
		}
	}
}

// Issue 13705: don't try to resolve onion addresses, etc
func TestAvoidDNSName(t *testing.T) {
	tests := []struct {
		name  string
		avoid bool
	}{
		{"foo.com", false},
		{"foo.com.", false},

		{"foo.onion.", true},
		{"foo.onion", true},
		{"foo.ONION", true},
		{"foo.ONION.", true},

		// But do resolve *.local address; Issue 16739
		{"foo.local.", false},
		{"foo.local", false},
		{"foo.LOCAL", false},
		{"foo.LOCAL.", false},

		{"", true}, // will be rejected earlier too

		// Without stuff before onion/local, they're fine to
		// use DNS. With a search path,
		// "onion.vegegtables.com" can use DNS. Without a
		// search path (or with a trailing dot), the queries
		// are just kinda useless, but don't reveal anything
		// private.
		{"local", false},
		{"onion", false},
		{"local.", false},
		{"onion.", false},
	}
	for _, tt := range tests {
		got := avoidDNS(tt.name)
		if got != tt.avoid {
			t.Errorf("avoidDNS(%q) = %v; want %v", tt.name, got, tt.avoid)
		}
	}
}

// Issue 13705: don't try to resolve onion addresses, etc
func TestLookupTorOnion(t *testing.T) {
	addrs, err := goLookupIP(context.Background(), "foo.onion")
	if len(addrs) > 0 {
		t.Errorf("unexpected addresses: %v", addrs)
	}
	if err != nil {
		t.Fatalf("lookup = %v; want nil", err)
	}
}

type resolvConfTest struct {
	dir  string
	path string
	*resolverConfig
}

func newResolvConfTest() (*resolvConfTest, error) {
	dir, err := ioutil.TempDir("", "go-resolvconftest")
	if err != nil {
		return nil, err
	}
	conf := &resolvConfTest{
		dir:            dir,
		path:           path.Join(dir, "resolv.conf"),
		resolverConfig: &resolvConf,
	}
	conf.initOnce.Do(conf.init)
	return conf, nil
}

func (conf *resolvConfTest) writeAndUpdate(lines []string) error {
	f, err := os.OpenFile(conf.path, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	if _, err := f.WriteString(strings.Join(lines, "\n")); err != nil {
		f.Close()
		return err
	}
	f.Close()
	if err := conf.forceUpdate(conf.path, time.Now().Add(time.Hour)); err != nil {
		return err
	}
	return nil
}

func (conf *resolvConfTest) forceUpdate(name string, lastChecked time.Time) error {
	dnsConf := dnsReadConfig(name)
	conf.mu.Lock()
	conf.dnsConfig = dnsConf
	conf.mu.Unlock()
	for i := 0; i < 5; i++ {
		if conf.tryAcquireSema() {
			conf.lastChecked = lastChecked
			conf.releaseSema()
			return nil
		}
	}
	return fmt.Errorf("tryAcquireSema for %s failed", name)
}

func (conf *resolvConfTest) servers() []string {
	conf.mu.RLock()
	servers := conf.dnsConfig.servers
	conf.mu.RUnlock()
	return servers
}

func (conf *resolvConfTest) teardown() error {
	err := conf.forceUpdate("/etc/resolv.conf", time.Time{})
	os.RemoveAll(conf.dir)
	return err
}

var updateResolvConfTests = []struct {
	name    string   // query name
	lines   []string // resolver configuration lines
	servers []string // expected name servers
}{
	{
		name:    "golang.org",
		lines:   []string{"nameserver 8.8.8.8"},
		servers: []string{"8.8.8.8:53"},
	},
	{
		name:    "",
		lines:   nil, // an empty resolv.conf should use defaultNS as name servers
		servers: defaultNS,
	},
	{
		name:    "www.example.com",
		lines:   []string{"nameserver 8.8.4.4"},
		servers: []string{"8.8.4.4:53"},
	},
}

func TestUpdateResolvConf(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	conf, err := newResolvConfTest()
	if err != nil {
		t.Fatal(err)
	}
	defer conf.teardown()

	for i, tt := range updateResolvConfTests {
		if err := conf.writeAndUpdate(tt.lines); err != nil {
			t.Error(err)
			continue
		}
		if tt.name != "" {
			var wg sync.WaitGroup
			const N = 10
			wg.Add(N)
			for j := 0; j < N; j++ {
				go func(name string) {
					defer wg.Done()
					ips, err := goLookupIP(context.Background(), name)
					if err != nil {
						t.Error(err)
						return
					}
					if len(ips) == 0 {
						t.Errorf("no records for %s", name)
						return
					}
				}(tt.name)
			}
			wg.Wait()
		}
		servers := conf.servers()
		if !reflect.DeepEqual(servers, tt.servers) {
			t.Errorf("#%d: got %v; want %v", i, servers, tt.servers)
			continue
		}
	}
}

var goLookupIPWithResolverConfigTests = []struct {
	name  string
	lines []string // resolver configuration lines
	error
	a, aaaa bool // whether response contains A, AAAA-record
}{
	// no records, transport timeout
	{
		"jgahvsekduiv9bw4b3qhn4ykdfgj0493iohkrjfhdvhjiu4j",
		[]string{
			"options timeout:1 attempts:1",
			"nameserver 255.255.255.255", // please forgive us for abuse of limited broadcast address
		},
		&DNSError{Name: "jgahvsekduiv9bw4b3qhn4ykdfgj0493iohkrjfhdvhjiu4j", Server: "255.255.255.255:53", IsTimeout: true},
		false, false,
	},

	// no records, non-existent domain
	{
		"jgahvsekduiv9bw4b3qhn4ykdfgj0493iohkrjfhdvhjiu4j",
		[]string{
			"options timeout:3 attempts:1",
			"nameserver 8.8.8.8",
		},
		&DNSError{Name: "jgahvsekduiv9bw4b3qhn4ykdfgj0493iohkrjfhdvhjiu4j", Server: "8.8.8.8:53", IsTimeout: false},
		false, false,
	},

	// a few A records, no AAAA records
	{
		"ipv4.google.com.",
		[]string{
			"nameserver 8.8.8.8",
			"nameserver 2001:4860:4860::8888",
		},
		nil,
		true, false,
	},
	{
		"ipv4.google.com",
		[]string{
			"domain golang.org",
			"nameserver 2001:4860:4860::8888",
			"nameserver 8.8.8.8",
		},
		nil,
		true, false,
	},
	{
		"ipv4.google.com",
		[]string{
			"search x.golang.org y.golang.org",
			"nameserver 2001:4860:4860::8888",
			"nameserver 8.8.8.8",
		},
		nil,
		true, false,
	},

	// no A records, a few AAAA records
	{
		"ipv6.google.com.",
		[]string{
			"nameserver 2001:4860:4860::8888",
			"nameserver 8.8.8.8",
		},
		nil,
		false, true,
	},
	{
		"ipv6.google.com",
		[]string{
			"domain golang.org",
			"nameserver 8.8.8.8",
			"nameserver 2001:4860:4860::8888",
		},
		nil,
		false, true,
	},
	{
		"ipv6.google.com",
		[]string{
			"search x.golang.org y.golang.org",
			"nameserver 8.8.8.8",
			"nameserver 2001:4860:4860::8888",
		},
		nil,
		false, true,
	},

	// both A and AAAA records
	{
		"hostname.as112.net", // see RFC 7534
		[]string{
			"domain golang.org",
			"nameserver 2001:4860:4860::8888",
			"nameserver 8.8.8.8",
		},
		nil,
		true, true,
	},
	{
		"hostname.as112.net", // see RFC 7534
		[]string{
			"search x.golang.org y.golang.org",
			"nameserver 2001:4860:4860::8888",
			"nameserver 8.8.8.8",
		},
		nil,
		true, true,
	},
}

func TestGoLookupIPWithResolverConfig(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	conf, err := newResolvConfTest()
	if err != nil {
		t.Fatal(err)
	}
	defer conf.teardown()

	for _, tt := range goLookupIPWithResolverConfigTests {
		if err := conf.writeAndUpdate(tt.lines); err != nil {
			t.Error(err)
			continue
		}
		addrs, err := goLookupIP(context.Background(), tt.name)
		if err != nil {
			// This test uses external network connectivity.
			// We need to take care with errors on both
			// DNS message exchange layer and DNS
			// transport layer because goLookupIP may fail
			// when the IP connectivty on node under test
			// gets lost during its run.
			if err, ok := err.(*DNSError); !ok || tt.error != nil && (err.Name != tt.error.(*DNSError).Name || err.Server != tt.error.(*DNSError).Server || err.IsTimeout != tt.error.(*DNSError).IsTimeout) {
				t.Errorf("got %v; want %v", err, tt.error)
			}
			continue
		}
		if len(addrs) == 0 {
			t.Errorf("no records for %s", tt.name)
		}
		if !tt.a && !tt.aaaa && len(addrs) > 0 {
			t.Errorf("unexpected %v for %s", addrs, tt.name)
		}
		for _, addr := range addrs {
			if !tt.a && addr.IP.To4() != nil {
				t.Errorf("got %v; must not be IPv4 address", addr)
			}
			if !tt.aaaa && addr.IP.To16() != nil && addr.IP.To4() == nil {
				t.Errorf("got %v; must not be IPv6 address", addr)
			}
		}
	}
}

// Test that goLookupIPOrder falls back to the host file when no DNS servers are available.
func TestGoLookupIPOrderFallbackToFile(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	// Add a config that simulates no dns servers being available.
	conf, err := newResolvConfTest()
	if err != nil {
		t.Fatal(err)
	}
	if err := conf.writeAndUpdate([]string{}); err != nil {
		t.Fatal(err)
	}
	// Redirect host file lookups.
	defer func(orig string) { testHookHostsPath = orig }(testHookHostsPath)
	testHookHostsPath = "testdata/hosts"

	for _, order := range []hostLookupOrder{hostLookupFilesDNS, hostLookupDNSFiles} {
		name := fmt.Sprintf("order %v", order)

		// First ensure that we get an error when contacting a non-existent host.
		_, err := goLookupIPOrder(context.Background(), "notarealhost", order)
		if err == nil {
			t.Errorf("%s: expected error while looking up name not in hosts file", name)
			continue
		}

		// Now check that we get an address when the name appears in the hosts file.
		addrs, err := goLookupIPOrder(context.Background(), "thor", order) // entry is in "testdata/hosts"
		if err != nil {
			t.Errorf("%s: expected to successfully lookup host entry", name)
			continue
		}
		if len(addrs) != 1 {
			t.Errorf("%s: expected exactly one result, but got %v", name, addrs)
			continue
		}
		if got, want := addrs[0].String(), "127.1.1.1"; got != want {
			t.Errorf("%s: address doesn't match expectation. got %v, want %v", name, got, want)
		}
	}
	defer conf.teardown()
}

// Issue 12712.
// When using search domains, return the error encountered
// querying the original name instead of an error encountered
// querying a generated name.
func TestErrorForOriginalNameWhenSearching(t *testing.T) {
	const fqdn = "doesnotexist.domain"

	origTestHookDNSDialer := testHookDNSDialer
	defer func() { testHookDNSDialer = origTestHookDNSDialer }()

	conf, err := newResolvConfTest()
	if err != nil {
		t.Fatal(err)
	}
	defer conf.teardown()

	if err := conf.writeAndUpdate([]string{"search servfail"}); err != nil {
		t.Fatal(err)
	}

	d := &fakeDNSDialer{}
	testHookDNSDialer = func() dnsDialer { return d }

	d.rh = func(s string, q *dnsMsg, _ time.Time) (*dnsMsg, error) {
		r := &dnsMsg{
			dnsMsgHdr: dnsMsgHdr{
				id: q.id,
			},
		}

		switch q.question[0].Name {
		case fqdn + ".servfail.":
			r.rcode = dnsRcodeServerFailure
		default:
			r.rcode = dnsRcodeNameError
		}

		return r, nil
	}

	_, err = goLookupIP(context.Background(), fqdn)
	if err == nil {
		t.Fatal("expected an error")
	}

	want := &DNSError{Name: fqdn, Err: errNoSuchHost.Error()}
	if err, ok := err.(*DNSError); !ok || err.Name != want.Name || err.Err != want.Err {
		t.Errorf("got %v; want %v", err, want)
	}
}

// Issue 15434. If a name server gives a lame referral, continue to the next.
func TestIgnoreLameReferrals(t *testing.T) {
	origTestHookDNSDialer := testHookDNSDialer
	defer func() { testHookDNSDialer = origTestHookDNSDialer }()

	conf, err := newResolvConfTest()
	if err != nil {
		t.Fatal(err)
	}
	defer conf.teardown()

	if err := conf.writeAndUpdate([]string{"nameserver 192.0.2.1", // the one that will give a lame referral
		"nameserver 192.0.2.2"}); err != nil {
		t.Fatal(err)
	}

	d := &fakeDNSDialer{}
	testHookDNSDialer = func() dnsDialer { return d }

	d.rh = func(s string, q *dnsMsg, _ time.Time) (*dnsMsg, error) {
		t.Log(s, q)
		r := &dnsMsg{
			dnsMsgHdr: dnsMsgHdr{
				id:       q.id,
				response: true,
			},
			question: q.question,
		}

		if s == "192.0.2.2:53" {
			r.recursion_available = true
			if q.question[0].Qtype == dnsTypeA {
				r.answer = []dnsRR{
					&dnsRR_A{
						Hdr: dnsRR_Header{
							Name:     q.question[0].Name,
							Rrtype:   dnsTypeA,
							Class:    dnsClassINET,
							Rdlength: 4,
						},
						A: TestAddr,
					},
				}
			}
		}

		return r, nil
	}

	addrs, err := goLookupIP(context.Background(), "www.golang.org")
	if err != nil {
		t.Fatal(err)
	}

	if got := len(addrs); got != 1 {
		t.Fatalf("got %d addresses, want 1", got)
	}

	if got, want := addrs[0].String(), "192.0.2.1"; got != want {
		t.Fatalf("got address %v, want %v", got, want)
	}
}

func BenchmarkGoLookupIP(b *testing.B) {
	testHookUninstaller.Do(uninstallTestHooks)
	ctx := context.Background()

	for i := 0; i < b.N; i++ {
		goLookupIP(ctx, "www.example.com")
	}
}

func BenchmarkGoLookupIPNoSuchHost(b *testing.B) {
	testHookUninstaller.Do(uninstallTestHooks)
	ctx := context.Background()

	for i := 0; i < b.N; i++ {
		goLookupIP(ctx, "some.nonexistent")
	}
}

func BenchmarkGoLookupIPWithBrokenNameServer(b *testing.B) {
	testHookUninstaller.Do(uninstallTestHooks)

	conf, err := newResolvConfTest()
	if err != nil {
		b.Fatal(err)
	}
	defer conf.teardown()

	lines := []string{
		"nameserver 203.0.113.254", // use TEST-NET-3 block, see RFC 5737
		"nameserver 8.8.8.8",
	}
	if err := conf.writeAndUpdate(lines); err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()

	for i := 0; i < b.N; i++ {
		goLookupIP(ctx, "www.example.com")
	}
}

type fakeDNSDialer struct {
	// reply handler
	rh func(s string, q *dnsMsg, t time.Time) (*dnsMsg, error)
}

func (f *fakeDNSDialer) dialDNS(_ context.Context, n, s string) (dnsConn, error) {
	return &fakeDNSConn{f.rh, s, time.Time{}}, nil
}

type fakeDNSConn struct {
	rh func(s string, q *dnsMsg, t time.Time) (*dnsMsg, error)
	s  string
	t  time.Time
}

func (f *fakeDNSConn) Close() error {
	return nil
}

func (f *fakeDNSConn) SetDeadline(t time.Time) error {
	f.t = t
	return nil
}

func (f *fakeDNSConn) dnsRoundTrip(q *dnsMsg) (*dnsMsg, error) {
	return f.rh(f.s, q, f.t)
}

// UDP round-tripper algorithm should ignore invalid DNS responses (issue 13281).
func TestIgnoreDNSForgeries(t *testing.T) {
	c, s := Pipe()
	go func() {
		b := make([]byte, 512)
		n, err := s.Read(b)
		if err != nil {
			t.Fatal(err)
		}

		msg := &dnsMsg{}
		if !msg.Unpack(b[:n]) {
			t.Fatal("invalid DNS query")
		}

		s.Write([]byte("garbage DNS response packet"))

		msg.response = true
		msg.id++ // make invalid ID
		b, ok := msg.Pack()
		if !ok {
			t.Fatal("failed to pack DNS response")
		}
		s.Write(b)

		msg.id-- // restore original ID
		msg.answer = []dnsRR{
			&dnsRR_A{
				Hdr: dnsRR_Header{
					Name:     "www.example.com.",
					Rrtype:   dnsTypeA,
					Class:    dnsClassINET,
					Rdlength: 4,
				},
				A: TestAddr,
			},
		}

		b, ok = msg.Pack()
		if !ok {
			t.Fatal("failed to pack DNS response")
		}
		s.Write(b)
	}()

	msg := &dnsMsg{
		dnsMsgHdr: dnsMsgHdr{
			id: 42,
		},
		question: []dnsQuestion{
			{
				Name:   "www.example.com.",
				Qtype:  dnsTypeA,
				Qclass: dnsClassINET,
			},
		},
	}

	resp, err := dnsRoundTripUDP(c, msg)
	if err != nil {
		t.Fatalf("dnsRoundTripUDP failed: %v", err)
	}

	if got := resp.answer[0].(*dnsRR_A).A; got != TestAddr {
		t.Errorf("got address %v, want %v", got, TestAddr)
	}
}

// Issue 16865. If a name server times out, continue to the next.
func TestRetryTimeout(t *testing.T) {
	origTestHookDNSDialer := testHookDNSDialer
	defer func() { testHookDNSDialer = origTestHookDNSDialer }()

	conf, err := newResolvConfTest()
	if err != nil {
		t.Fatal(err)
	}
	defer conf.teardown()

	if err := conf.writeAndUpdate([]string{"nameserver 192.0.2.1", // the one that will timeout
		"nameserver 192.0.2.2"}); err != nil {
		t.Fatal(err)
	}

	d := &fakeDNSDialer{}
	testHookDNSDialer = func() dnsDialer { return d }

	var deadline0 time.Time

	d.rh = func(s string, q *dnsMsg, deadline time.Time) (*dnsMsg, error) {
		t.Log(s, q, deadline)

		if deadline.IsZero() {
			t.Error("zero deadline")
		}

		if s == "192.0.2.1:53" {
			deadline0 = deadline
			time.Sleep(10 * time.Millisecond)
			return nil, errTimeout
		}

		if deadline == deadline0 {
			t.Error("deadline didn't change")
		}

		r := &dnsMsg{
			dnsMsgHdr: dnsMsgHdr{
				id:                  q.id,
				response:            true,
				recursion_available: true,
			},
			question: q.question,
			answer: []dnsRR{
				&dnsRR_CNAME{
					Hdr: dnsRR_Header{
						Name:   q.question[0].Name,
						Rrtype: dnsTypeCNAME,
						Class:  dnsClassINET,
					},
					Cname: "golang.org",
				},
			},
		}
		return r, nil
	}

	_, err = goLookupCNAME(context.Background(), "www.golang.org")
	if err != nil {
		t.Fatal(err)
	}

	if deadline0.IsZero() {
		t.Error("deadline0 still zero", deadline0)
	}
}

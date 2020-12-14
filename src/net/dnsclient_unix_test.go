// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris

package net

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"golang.org/x/net/dns/dnsmessage"
)

var goResolver = Resolver{PreferGo: true}

// Test address from 192.0.2.0/24 block, reserved by RFC 5737 for documentation.
var TestAddr = [4]byte{0xc0, 0x00, 0x02, 0x01}

// Test address from 2001:db8::/32 block, reserved by RFC 3849 for documentation.
var TestAddr6 = [16]byte{0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}

func mustNewName(name string) dnsmessage.Name {
	nn, err := dnsmessage.NewName(name)
	if err != nil {
		panic(fmt.Sprint("creating name: ", err))
	}
	return nn
}

func mustQuestion(name string, qtype dnsmessage.Type, class dnsmessage.Class) dnsmessage.Question {
	return dnsmessage.Question{
		Name:  mustNewName(name),
		Type:  qtype,
		Class: class,
	}
}

var dnsTransportFallbackTests = []struct {
	server   string
	question dnsmessage.Question
	timeout  int
	rcode    dnsmessage.RCode
}{
	// Querying "com." with qtype=255 usually makes an answer
	// which requires more than 512 bytes.
	{"8.8.8.8:53", mustQuestion("com.", dnsmessage.TypeALL, dnsmessage.ClassINET), 2, dnsmessage.RCodeSuccess},
	{"8.8.4.4:53", mustQuestion("com.", dnsmessage.TypeALL, dnsmessage.ClassINET), 4, dnsmessage.RCodeSuccess},
}

func TestDNSTransportFallback(t *testing.T) {
	fake := fakeDNSServer{
		rh: func(n, _ string, q dnsmessage.Message, _ time.Time) (dnsmessage.Message, error) {
			r := dnsmessage.Message{
				Header: dnsmessage.Header{
					ID:       q.Header.ID,
					Response: true,
					RCode:    dnsmessage.RCodeSuccess,
				},
				Questions: q.Questions,
			}
			if n == "udp" {
				r.Header.Truncated = true
			}
			return r, nil
		},
	}
	r := Resolver{PreferGo: true, Dial: fake.DialContext}
	for _, tt := range dnsTransportFallbackTests {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		_, h, err := r.exchange(ctx, tt.server, tt.question, time.Second, useUDPOrTCP)
		if err != nil {
			t.Error(err)
			continue
		}
		if h.RCode != tt.rcode {
			t.Errorf("got %v from %v; want %v", h.RCode, tt.server, tt.rcode)
			continue
		}
	}
}

// See RFC 6761 for further information about the reserved, pseudo
// domain names.
var specialDomainNameTests = []struct {
	question dnsmessage.Question
	rcode    dnsmessage.RCode
}{
	// Name resolution APIs and libraries should not recognize the
	// followings as special.
	{mustQuestion("1.0.168.192.in-addr.arpa.", dnsmessage.TypePTR, dnsmessage.ClassINET), dnsmessage.RCodeNameError},
	{mustQuestion("test.", dnsmessage.TypeALL, dnsmessage.ClassINET), dnsmessage.RCodeNameError},
	{mustQuestion("example.com.", dnsmessage.TypeALL, dnsmessage.ClassINET), dnsmessage.RCodeSuccess},

	// Name resolution APIs and libraries should recognize the
	// followings as special and should not send any queries.
	// Though, we test those names here for verifying negative
	// answers at DNS query-response interaction level.
	{mustQuestion("localhost.", dnsmessage.TypeALL, dnsmessage.ClassINET), dnsmessage.RCodeNameError},
	{mustQuestion("invalid.", dnsmessage.TypeALL, dnsmessage.ClassINET), dnsmessage.RCodeNameError},
}

func TestSpecialDomainName(t *testing.T) {
	fake := fakeDNSServer{rh: func(_, _ string, q dnsmessage.Message, _ time.Time) (dnsmessage.Message, error) {
		r := dnsmessage.Message{
			Header: dnsmessage.Header{
				ID:       q.ID,
				Response: true,
			},
			Questions: q.Questions,
		}

		switch q.Questions[0].Name.String() {
		case "example.com.":
			r.Header.RCode = dnsmessage.RCodeSuccess
		default:
			r.Header.RCode = dnsmessage.RCodeNameError
		}

		return r, nil
	}}
	r := Resolver{PreferGo: true, Dial: fake.DialContext}
	server := "8.8.8.8:53"
	for _, tt := range specialDomainNameTests {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		_, h, err := r.exchange(ctx, server, tt.question, 3*time.Second, useUDPOrTCP)
		if err != nil {
			t.Error(err)
			continue
		}
		if h.RCode != tt.rcode {
			t.Errorf("got %v from %v; want %v", h.RCode, server, tt.rcode)
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
		// "onion.vegetables.com" can use DNS. Without a
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

var fakeDNSServerSuccessful = fakeDNSServer{rh: func(_, _ string, q dnsmessage.Message, _ time.Time) (dnsmessage.Message, error) {
	r := dnsmessage.Message{
		Header: dnsmessage.Header{
			ID:       q.ID,
			Response: true,
		},
		Questions: q.Questions,
	}
	if len(q.Questions) == 1 && q.Questions[0].Type == dnsmessage.TypeA {
		r.Answers = []dnsmessage.Resource{
			{
				Header: dnsmessage.ResourceHeader{
					Name:   q.Questions[0].Name,
					Type:   dnsmessage.TypeA,
					Class:  dnsmessage.ClassINET,
					Length: 4,
				},
				Body: &dnsmessage.AResource{
					A: TestAddr,
				},
			},
		}
	}
	return r, nil
}}

// Issue 13705: don't try to resolve onion addresses, etc
func TestLookupTorOnion(t *testing.T) {
	defer dnsWaitGroup.Wait()
	r := Resolver{PreferGo: true, Dial: fakeDNSServerSuccessful.DialContext}
	addrs, err := r.LookupIPAddr(context.Background(), "foo.onion")
	if err != nil {
		t.Fatalf("lookup = %v; want nil", err)
	}
	if len(addrs) > 0 {
		t.Errorf("unexpected addresses: %v", addrs)
	}
}

type resolvConfTest struct {
	dir  string
	path string
	*resolverConfig
}

func newResolvConfTest() (*resolvConfTest, error) {
	dir, err := os.MkdirTemp("", "go-resolvconftest")
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
	defer dnsWaitGroup.Wait()

	r := Resolver{PreferGo: true, Dial: fakeDNSServerSuccessful.DialContext}

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
					ips, err := r.LookupIPAddr(context.Background(), name)
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
	defer dnsWaitGroup.Wait()
	fake := fakeDNSServer{rh: func(n, s string, q dnsmessage.Message, _ time.Time) (dnsmessage.Message, error) {
		switch s {
		case "[2001:4860:4860::8888]:53", "8.8.8.8:53":
			break
		default:
			time.Sleep(10 * time.Millisecond)
			return dnsmessage.Message{}, os.ErrDeadlineExceeded
		}
		r := dnsmessage.Message{
			Header: dnsmessage.Header{
				ID:       q.ID,
				Response: true,
			},
			Questions: q.Questions,
		}
		for _, question := range q.Questions {
			switch question.Type {
			case dnsmessage.TypeA:
				switch question.Name.String() {
				case "hostname.as112.net.":
					break
				case "ipv4.google.com.":
					r.Answers = append(r.Answers, dnsmessage.Resource{
						Header: dnsmessage.ResourceHeader{
							Name:   q.Questions[0].Name,
							Type:   dnsmessage.TypeA,
							Class:  dnsmessage.ClassINET,
							Length: 4,
						},
						Body: &dnsmessage.AResource{
							A: TestAddr,
						},
					})
				default:

				}
			case dnsmessage.TypeAAAA:
				switch question.Name.String() {
				case "hostname.as112.net.":
					break
				case "ipv6.google.com.":
					r.Answers = append(r.Answers, dnsmessage.Resource{
						Header: dnsmessage.ResourceHeader{
							Name:   q.Questions[0].Name,
							Type:   dnsmessage.TypeAAAA,
							Class:  dnsmessage.ClassINET,
							Length: 16,
						},
						Body: &dnsmessage.AAAAResource{
							AAAA: TestAddr6,
						},
					})
				}
			}
		}
		return r, nil
	}}
	r := Resolver{PreferGo: true, Dial: fake.DialContext}

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
		addrs, err := r.LookupIPAddr(context.Background(), tt.name)
		if err != nil {
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
	defer dnsWaitGroup.Wait()

	fake := fakeDNSServer{rh: func(n, s string, q dnsmessage.Message, tm time.Time) (dnsmessage.Message, error) {
		r := dnsmessage.Message{
			Header: dnsmessage.Header{
				ID:       q.ID,
				Response: true,
			},
			Questions: q.Questions,
		}
		return r, nil
	}}
	r := Resolver{PreferGo: true, Dial: fake.DialContext}

	// Add a config that simulates no dns servers being available.
	conf, err := newResolvConfTest()
	if err != nil {
		t.Fatal(err)
	}
	defer conf.teardown()

	if err := conf.writeAndUpdate([]string{}); err != nil {
		t.Fatal(err)
	}
	// Redirect host file lookups.
	defer func(orig string) { testHookHostsPath = orig }(testHookHostsPath)
	testHookHostsPath = "testdata/hosts"

	for _, order := range []hostLookupOrder{hostLookupFilesDNS, hostLookupDNSFiles} {
		name := fmt.Sprintf("order %v", order)

		// First ensure that we get an error when contacting a non-existent host.
		_, _, err := r.goLookupIPCNAMEOrder(context.Background(), "notarealhost", order)
		if err == nil {
			t.Errorf("%s: expected error while looking up name not in hosts file", name)
			continue
		}

		// Now check that we get an address when the name appears in the hosts file.
		addrs, _, err := r.goLookupIPCNAMEOrder(context.Background(), "thor", order) // entry is in "testdata/hosts"
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
}

// Issue 12712.
// When using search domains, return the error encountered
// querying the original name instead of an error encountered
// querying a generated name.
func TestErrorForOriginalNameWhenSearching(t *testing.T) {
	defer dnsWaitGroup.Wait()

	const fqdn = "doesnotexist.domain"

	conf, err := newResolvConfTest()
	if err != nil {
		t.Fatal(err)
	}
	defer conf.teardown()

	if err := conf.writeAndUpdate([]string{"search servfail"}); err != nil {
		t.Fatal(err)
	}

	fake := fakeDNSServer{rh: func(_, _ string, q dnsmessage.Message, _ time.Time) (dnsmessage.Message, error) {
		r := dnsmessage.Message{
			Header: dnsmessage.Header{
				ID:       q.ID,
				Response: true,
			},
			Questions: q.Questions,
		}

		switch q.Questions[0].Name.String() {
		case fqdn + ".servfail.":
			r.Header.RCode = dnsmessage.RCodeServerFailure
		default:
			r.Header.RCode = dnsmessage.RCodeNameError
		}

		return r, nil
	}}

	cases := []struct {
		strictErrors bool
		wantErr      *DNSError
	}{
		{true, &DNSError{Name: fqdn, Err: "server misbehaving", IsTemporary: true}},
		{false, &DNSError{Name: fqdn, Err: errNoSuchHost.Error(), IsNotFound: true}},
	}
	for _, tt := range cases {
		r := Resolver{PreferGo: true, StrictErrors: tt.strictErrors, Dial: fake.DialContext}
		_, err = r.LookupIPAddr(context.Background(), fqdn)
		if err == nil {
			t.Fatal("expected an error")
		}

		want := tt.wantErr
		if err, ok := err.(*DNSError); !ok || err.Name != want.Name || err.Err != want.Err || err.IsTemporary != want.IsTemporary {
			t.Errorf("got %v; want %v", err, want)
		}
	}
}

// Issue 15434. If a name server gives a lame referral, continue to the next.
func TestIgnoreLameReferrals(t *testing.T) {
	defer dnsWaitGroup.Wait()

	conf, err := newResolvConfTest()
	if err != nil {
		t.Fatal(err)
	}
	defer conf.teardown()

	if err := conf.writeAndUpdate([]string{"nameserver 192.0.2.1", // the one that will give a lame referral
		"nameserver 192.0.2.2"}); err != nil {
		t.Fatal(err)
	}

	fake := fakeDNSServer{rh: func(_, s string, q dnsmessage.Message, _ time.Time) (dnsmessage.Message, error) {
		t.Log(s, q)
		r := dnsmessage.Message{
			Header: dnsmessage.Header{
				ID:       q.ID,
				Response: true,
			},
			Questions: q.Questions,
		}

		if s == "192.0.2.2:53" {
			r.Header.RecursionAvailable = true
			if q.Questions[0].Type == dnsmessage.TypeA {
				r.Answers = []dnsmessage.Resource{
					{
						Header: dnsmessage.ResourceHeader{
							Name:   q.Questions[0].Name,
							Type:   dnsmessage.TypeA,
							Class:  dnsmessage.ClassINET,
							Length: 4,
						},
						Body: &dnsmessage.AResource{
							A: TestAddr,
						},
					},
				}
			}
		}

		return r, nil
	}}
	r := Resolver{PreferGo: true, Dial: fake.DialContext}

	addrs, err := r.LookupIPAddr(context.Background(), "www.golang.org")
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
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		goResolver.LookupIPAddr(ctx, "www.example.com")
	}
}

func BenchmarkGoLookupIPNoSuchHost(b *testing.B) {
	testHookUninstaller.Do(uninstallTestHooks)
	ctx := context.Background()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		goResolver.LookupIPAddr(ctx, "some.nonexistent")
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
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		goResolver.LookupIPAddr(ctx, "www.example.com")
	}
}

type fakeDNSServer struct {
	rh        func(n, s string, q dnsmessage.Message, t time.Time) (dnsmessage.Message, error)
	alwaysTCP bool
}

func (server *fakeDNSServer) DialContext(_ context.Context, n, s string) (Conn, error) {
	if server.alwaysTCP || n == "tcp" || n == "tcp4" || n == "tcp6" {
		return &fakeDNSConn{tcp: true, server: server, n: n, s: s}, nil
	}
	return &fakeDNSPacketConn{fakeDNSConn: fakeDNSConn{tcp: false, server: server, n: n, s: s}}, nil
}

type fakeDNSConn struct {
	Conn
	tcp    bool
	server *fakeDNSServer
	n      string
	s      string
	q      dnsmessage.Message
	t      time.Time
	buf    []byte
}

func (f *fakeDNSConn) Close() error {
	return nil
}

func (f *fakeDNSConn) Read(b []byte) (int, error) {
	if len(f.buf) > 0 {
		n := copy(b, f.buf)
		f.buf = f.buf[n:]
		return n, nil
	}

	resp, err := f.server.rh(f.n, f.s, f.q, f.t)
	if err != nil {
		return 0, err
	}

	bb := make([]byte, 2, 514)
	bb, err = resp.AppendPack(bb)
	if err != nil {
		return 0, fmt.Errorf("cannot marshal DNS message: %v", err)
	}

	if f.tcp {
		l := len(bb) - 2
		bb[0] = byte(l >> 8)
		bb[1] = byte(l)
		f.buf = bb
		return f.Read(b)
	}

	bb = bb[2:]
	if len(b) < len(bb) {
		return 0, errors.New("read would fragment DNS message")
	}

	copy(b, bb)
	return len(bb), nil
}

func (f *fakeDNSConn) Write(b []byte) (int, error) {
	if f.tcp && len(b) >= 2 {
		b = b[2:]
	}
	if f.q.Unpack(b) != nil {
		return 0, fmt.Errorf("cannot unmarshal DNS message fake %s (%d)", f.n, len(b))
	}
	return len(b), nil
}

func (f *fakeDNSConn) SetDeadline(t time.Time) error {
	f.t = t
	return nil
}

type fakeDNSPacketConn struct {
	PacketConn
	fakeDNSConn
}

func (f *fakeDNSPacketConn) SetDeadline(t time.Time) error {
	return f.fakeDNSConn.SetDeadline(t)
}

func (f *fakeDNSPacketConn) Close() error {
	return f.fakeDNSConn.Close()
}

// UDP round-tripper algorithm should ignore invalid DNS responses (issue 13281).
func TestIgnoreDNSForgeries(t *testing.T) {
	c, s := Pipe()
	go func() {
		b := make([]byte, 512)
		n, err := s.Read(b)
		if err != nil {
			t.Error(err)
			return
		}

		var msg dnsmessage.Message
		if msg.Unpack(b[:n]) != nil {
			t.Error("invalid DNS query:", err)
			return
		}

		s.Write([]byte("garbage DNS response packet"))

		msg.Header.Response = true
		msg.Header.ID++ // make invalid ID

		if b, err = msg.Pack(); err != nil {
			t.Error("failed to pack DNS response:", err)
			return
		}
		s.Write(b)

		msg.Header.ID-- // restore original ID
		msg.Answers = []dnsmessage.Resource{
			{
				Header: dnsmessage.ResourceHeader{
					Name:   mustNewName("www.example.com."),
					Type:   dnsmessage.TypeA,
					Class:  dnsmessage.ClassINET,
					Length: 4,
				},
				Body: &dnsmessage.AResource{
					A: TestAddr,
				},
			},
		}

		b, err = msg.Pack()
		if err != nil {
			t.Error("failed to pack DNS response:", err)
			return
		}
		s.Write(b)
	}()

	msg := dnsmessage.Message{
		Header: dnsmessage.Header{
			ID: 42,
		},
		Questions: []dnsmessage.Question{
			{
				Name:  mustNewName("www.example.com."),
				Type:  dnsmessage.TypeA,
				Class: dnsmessage.ClassINET,
			},
		},
	}

	b, err := msg.Pack()
	if err != nil {
		t.Fatal("Pack failed:", err)
	}

	p, _, err := dnsPacketRoundTrip(c, 42, msg.Questions[0], b)
	if err != nil {
		t.Fatalf("dnsPacketRoundTrip failed: %v", err)
	}

	p.SkipAllQuestions()
	as, err := p.AllAnswers()
	if err != nil {
		t.Fatal("AllAnswers failed:", err)
	}
	if got := as[0].Body.(*dnsmessage.AResource).A; got != TestAddr {
		t.Errorf("got address %v, want %v", got, TestAddr)
	}
}

// Issue 16865. If a name server times out, continue to the next.
func TestRetryTimeout(t *testing.T) {
	defer dnsWaitGroup.Wait()

	conf, err := newResolvConfTest()
	if err != nil {
		t.Fatal(err)
	}
	defer conf.teardown()

	testConf := []string{
		"nameserver 192.0.2.1", // the one that will timeout
		"nameserver 192.0.2.2",
	}
	if err := conf.writeAndUpdate(testConf); err != nil {
		t.Fatal(err)
	}

	var deadline0 time.Time

	fake := fakeDNSServer{rh: func(_, s string, q dnsmessage.Message, deadline time.Time) (dnsmessage.Message, error) {
		t.Log(s, q, deadline)

		if deadline.IsZero() {
			t.Error("zero deadline")
		}

		if s == "192.0.2.1:53" {
			deadline0 = deadline
			time.Sleep(10 * time.Millisecond)
			return dnsmessage.Message{}, os.ErrDeadlineExceeded
		}

		if deadline.Equal(deadline0) {
			t.Error("deadline didn't change")
		}

		return mockTXTResponse(q), nil
	}}
	r := &Resolver{PreferGo: true, Dial: fake.DialContext}

	_, err = r.LookupTXT(context.Background(), "www.golang.org")
	if err != nil {
		t.Fatal(err)
	}

	if deadline0.IsZero() {
		t.Error("deadline0 still zero", deadline0)
	}
}

func TestRotate(t *testing.T) {
	// without rotation, always uses the first server
	testRotate(t, false, []string{"192.0.2.1", "192.0.2.2"}, []string{"192.0.2.1:53", "192.0.2.1:53", "192.0.2.1:53"})

	// with rotation, rotates through back to first
	testRotate(t, true, []string{"192.0.2.1", "192.0.2.2"}, []string{"192.0.2.1:53", "192.0.2.2:53", "192.0.2.1:53"})
}

func testRotate(t *testing.T, rotate bool, nameservers, wantServers []string) {
	defer dnsWaitGroup.Wait()

	conf, err := newResolvConfTest()
	if err != nil {
		t.Fatal(err)
	}
	defer conf.teardown()

	var confLines []string
	for _, ns := range nameservers {
		confLines = append(confLines, "nameserver "+ns)
	}
	if rotate {
		confLines = append(confLines, "options rotate")
	}

	if err := conf.writeAndUpdate(confLines); err != nil {
		t.Fatal(err)
	}

	var usedServers []string
	fake := fakeDNSServer{rh: func(_, s string, q dnsmessage.Message, deadline time.Time) (dnsmessage.Message, error) {
		usedServers = append(usedServers, s)
		return mockTXTResponse(q), nil
	}}
	r := Resolver{PreferGo: true, Dial: fake.DialContext}

	// len(nameservers) + 1 to allow rotation to get back to start
	for i := 0; i < len(nameservers)+1; i++ {
		if _, err := r.LookupTXT(context.Background(), "www.golang.org"); err != nil {
			t.Fatal(err)
		}
	}

	if !reflect.DeepEqual(usedServers, wantServers) {
		t.Errorf("rotate=%t got used servers:\n%v\nwant:\n%v", rotate, usedServers, wantServers)
	}
}

func mockTXTResponse(q dnsmessage.Message) dnsmessage.Message {
	r := dnsmessage.Message{
		Header: dnsmessage.Header{
			ID:                 q.ID,
			Response:           true,
			RecursionAvailable: true,
		},
		Questions: q.Questions,
		Answers: []dnsmessage.Resource{
			{
				Header: dnsmessage.ResourceHeader{
					Name:  q.Questions[0].Name,
					Type:  dnsmessage.TypeTXT,
					Class: dnsmessage.ClassINET,
				},
				Body: &dnsmessage.TXTResource{
					TXT: []string{"ok"},
				},
			},
		},
	}

	return r
}

// Issue 17448. With StrictErrors enabled, temporary errors should make
// LookupIP fail rather than return a partial result.
func TestStrictErrorsLookupIP(t *testing.T) {
	defer dnsWaitGroup.Wait()

	conf, err := newResolvConfTest()
	if err != nil {
		t.Fatal(err)
	}
	defer conf.teardown()

	confData := []string{
		"nameserver 192.0.2.53",
		"search x.golang.org y.golang.org",
	}
	if err := conf.writeAndUpdate(confData); err != nil {
		t.Fatal(err)
	}

	const name = "test-issue19592"
	const server = "192.0.2.53:53"
	const searchX = "test-issue19592.x.golang.org."
	const searchY = "test-issue19592.y.golang.org."
	const ip4 = "192.0.2.1"
	const ip6 = "2001:db8::1"

	type resolveWhichEnum int
	const (
		resolveOK resolveWhichEnum = iota
		resolveOpError
		resolveServfail
		resolveTimeout
	)

	makeTempError := func(err string) error {
		return &DNSError{
			Err:         err,
			Name:        name,
			Server:      server,
			IsTemporary: true,
		}
	}
	makeTimeout := func() error {
		return &DNSError{
			Err:       os.ErrDeadlineExceeded.Error(),
			Name:      name,
			Server:    server,
			IsTimeout: true,
		}
	}
	makeNxDomain := func() error {
		return &DNSError{
			Err:        errNoSuchHost.Error(),
			Name:       name,
			Server:     server,
			IsNotFound: true,
		}
	}

	cases := []struct {
		desc          string
		resolveWhich  func(quest dnsmessage.Question) resolveWhichEnum
		wantStrictErr error
		wantLaxErr    error
		wantIPs       []string
	}{
		{
			desc: "No errors",
			resolveWhich: func(quest dnsmessage.Question) resolveWhichEnum {
				return resolveOK
			},
			wantIPs: []string{ip4, ip6},
		},
		{
			desc: "searchX error fails in strict mode",
			resolveWhich: func(quest dnsmessage.Question) resolveWhichEnum {
				if quest.Name.String() == searchX {
					return resolveTimeout
				}
				return resolveOK
			},
			wantStrictErr: makeTimeout(),
			wantIPs:       []string{ip4, ip6},
		},
		{
			desc: "searchX IPv4-only timeout fails in strict mode",
			resolveWhich: func(quest dnsmessage.Question) resolveWhichEnum {
				if quest.Name.String() == searchX && quest.Type == dnsmessage.TypeA {
					return resolveTimeout
				}
				return resolveOK
			},
			wantStrictErr: makeTimeout(),
			wantIPs:       []string{ip4, ip6},
		},
		{
			desc: "searchX IPv6-only servfail fails in strict mode",
			resolveWhich: func(quest dnsmessage.Question) resolveWhichEnum {
				if quest.Name.String() == searchX && quest.Type == dnsmessage.TypeAAAA {
					return resolveServfail
				}
				return resolveOK
			},
			wantStrictErr: makeTempError("server misbehaving"),
			wantIPs:       []string{ip4, ip6},
		},
		{
			desc: "searchY error always fails",
			resolveWhich: func(quest dnsmessage.Question) resolveWhichEnum {
				if quest.Name.String() == searchY {
					return resolveTimeout
				}
				return resolveOK
			},
			wantStrictErr: makeTimeout(),
			wantLaxErr:    makeNxDomain(), // This one reaches the "test." FQDN.
		},
		{
			desc: "searchY IPv4-only socket error fails in strict mode",
			resolveWhich: func(quest dnsmessage.Question) resolveWhichEnum {
				if quest.Name.String() == searchY && quest.Type == dnsmessage.TypeA {
					return resolveOpError
				}
				return resolveOK
			},
			wantStrictErr: makeTempError("write: socket on fire"),
			wantIPs:       []string{ip6},
		},
		{
			desc: "searchY IPv6-only timeout fails in strict mode",
			resolveWhich: func(quest dnsmessage.Question) resolveWhichEnum {
				if quest.Name.String() == searchY && quest.Type == dnsmessage.TypeAAAA {
					return resolveTimeout
				}
				return resolveOK
			},
			wantStrictErr: makeTimeout(),
			wantIPs:       []string{ip4},
		},
	}

	for i, tt := range cases {
		fake := fakeDNSServer{rh: func(_, s string, q dnsmessage.Message, deadline time.Time) (dnsmessage.Message, error) {
			t.Log(s, q)

			switch tt.resolveWhich(q.Questions[0]) {
			case resolveOK:
				// Handle below.
			case resolveOpError:
				return dnsmessage.Message{}, &OpError{Op: "write", Err: fmt.Errorf("socket on fire")}
			case resolveServfail:
				return dnsmessage.Message{
					Header: dnsmessage.Header{
						ID:       q.ID,
						Response: true,
						RCode:    dnsmessage.RCodeServerFailure,
					},
					Questions: q.Questions,
				}, nil
			case resolveTimeout:
				return dnsmessage.Message{}, os.ErrDeadlineExceeded
			default:
				t.Fatal("Impossible resolveWhich")
			}

			switch q.Questions[0].Name.String() {
			case searchX, name + ".":
				// Return NXDOMAIN to utilize the search list.
				return dnsmessage.Message{
					Header: dnsmessage.Header{
						ID:       q.ID,
						Response: true,
						RCode:    dnsmessage.RCodeNameError,
					},
					Questions: q.Questions,
				}, nil
			case searchY:
				// Return records below.
			default:
				return dnsmessage.Message{}, fmt.Errorf("Unexpected Name: %v", q.Questions[0].Name)
			}

			r := dnsmessage.Message{
				Header: dnsmessage.Header{
					ID:       q.ID,
					Response: true,
				},
				Questions: q.Questions,
			}
			switch q.Questions[0].Type {
			case dnsmessage.TypeA:
				r.Answers = []dnsmessage.Resource{
					{
						Header: dnsmessage.ResourceHeader{
							Name:   q.Questions[0].Name,
							Type:   dnsmessage.TypeA,
							Class:  dnsmessage.ClassINET,
							Length: 4,
						},
						Body: &dnsmessage.AResource{
							A: TestAddr,
						},
					},
				}
			case dnsmessage.TypeAAAA:
				r.Answers = []dnsmessage.Resource{
					{
						Header: dnsmessage.ResourceHeader{
							Name:   q.Questions[0].Name,
							Type:   dnsmessage.TypeAAAA,
							Class:  dnsmessage.ClassINET,
							Length: 16,
						},
						Body: &dnsmessage.AAAAResource{
							AAAA: TestAddr6,
						},
					},
				}
			default:
				return dnsmessage.Message{}, fmt.Errorf("Unexpected Type: %v", q.Questions[0].Type)
			}
			return r, nil
		}}

		for _, strict := range []bool{true, false} {
			r := Resolver{PreferGo: true, StrictErrors: strict, Dial: fake.DialContext}
			ips, err := r.LookupIPAddr(context.Background(), name)

			var wantErr error
			if strict {
				wantErr = tt.wantStrictErr
			} else {
				wantErr = tt.wantLaxErr
			}
			if !reflect.DeepEqual(err, wantErr) {
				t.Errorf("#%d (%s) strict=%v: got err %#v; want %#v", i, tt.desc, strict, err, wantErr)
			}

			gotIPs := map[string]struct{}{}
			for _, ip := range ips {
				gotIPs[ip.String()] = struct{}{}
			}
			wantIPs := map[string]struct{}{}
			if wantErr == nil {
				for _, ip := range tt.wantIPs {
					wantIPs[ip] = struct{}{}
				}
			}
			if !reflect.DeepEqual(gotIPs, wantIPs) {
				t.Errorf("#%d (%s) strict=%v: got ips %v; want %v", i, tt.desc, strict, gotIPs, wantIPs)
			}
		}
	}
}

// Issue 17448. With StrictErrors enabled, temporary errors should make
// LookupTXT stop walking the search list.
func TestStrictErrorsLookupTXT(t *testing.T) {
	defer dnsWaitGroup.Wait()

	conf, err := newResolvConfTest()
	if err != nil {
		t.Fatal(err)
	}
	defer conf.teardown()

	confData := []string{
		"nameserver 192.0.2.53",
		"search x.golang.org y.golang.org",
	}
	if err := conf.writeAndUpdate(confData); err != nil {
		t.Fatal(err)
	}

	const name = "test"
	const server = "192.0.2.53:53"
	const searchX = "test.x.golang.org."
	const searchY = "test.y.golang.org."
	const txt = "Hello World"

	fake := fakeDNSServer{rh: func(_, s string, q dnsmessage.Message, deadline time.Time) (dnsmessage.Message, error) {
		t.Log(s, q)

		switch q.Questions[0].Name.String() {
		case searchX:
			return dnsmessage.Message{}, os.ErrDeadlineExceeded
		case searchY:
			return mockTXTResponse(q), nil
		default:
			return dnsmessage.Message{}, fmt.Errorf("Unexpected Name: %v", q.Questions[0].Name)
		}
	}}

	for _, strict := range []bool{true, false} {
		r := Resolver{StrictErrors: strict, Dial: fake.DialContext}
		p, _, err := r.lookup(context.Background(), name, dnsmessage.TypeTXT)
		var wantErr error
		var wantRRs int
		if strict {
			wantErr = &DNSError{
				Err:       os.ErrDeadlineExceeded.Error(),
				Name:      name,
				Server:    server,
				IsTimeout: true,
			}
		} else {
			wantRRs = 1
		}
		if !reflect.DeepEqual(err, wantErr) {
			t.Errorf("strict=%v: got err %#v; want %#v", strict, err, wantErr)
		}
		a, err := p.AllAnswers()
		if err != nil {
			a = nil
		}
		if len(a) != wantRRs {
			t.Errorf("strict=%v: got %v; want %v", strict, len(a), wantRRs)
		}
	}
}

// Test for a race between uninstalling the test hooks and closing a
// socket connection. This used to fail when testing with -race.
func TestDNSGoroutineRace(t *testing.T) {
	defer dnsWaitGroup.Wait()

	fake := fakeDNSServer{rh: func(n, s string, q dnsmessage.Message, t time.Time) (dnsmessage.Message, error) {
		time.Sleep(10 * time.Microsecond)
		return dnsmessage.Message{}, os.ErrDeadlineExceeded
	}}
	r := Resolver{PreferGo: true, Dial: fake.DialContext}

	// The timeout here is less than the timeout used by the server,
	// so the goroutine started to query the (fake) server will hang
	// around after this test is done if we don't call dnsWaitGroup.Wait.
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Microsecond)
	defer cancel()
	_, err := r.LookupIPAddr(ctx, "where.are.they.now")
	if err == nil {
		t.Fatal("fake DNS lookup unexpectedly succeeded")
	}
}

func lookupWithFake(fake fakeDNSServer, name string, typ dnsmessage.Type) error {
	r := Resolver{PreferGo: true, Dial: fake.DialContext}

	resolvConf.mu.RLock()
	conf := resolvConf.dnsConfig
	resolvConf.mu.RUnlock()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	_, _, err := r.tryOneName(ctx, conf, name, typ)
	return err
}

// Issue 8434: verify that Temporary returns true on an error when rcode
// is SERVFAIL
func TestIssue8434(t *testing.T) {
	err := lookupWithFake(fakeDNSServer{
		rh: func(n, _ string, q dnsmessage.Message, _ time.Time) (dnsmessage.Message, error) {
			return dnsmessage.Message{
				Header: dnsmessage.Header{
					ID:       q.ID,
					Response: true,
					RCode:    dnsmessage.RCodeServerFailure,
				},
				Questions: q.Questions,
			}, nil
		},
	}, "golang.org.", dnsmessage.TypeALL)
	if err == nil {
		t.Fatal("expected an error")
	}
	if ne, ok := err.(Error); !ok {
		t.Fatalf("err = %#v; wanted something supporting net.Error", err)
	} else if !ne.Temporary() {
		t.Fatalf("Temporary = false for err = %#v; want Temporary == true", err)
	}
	if de, ok := err.(*DNSError); !ok {
		t.Fatalf("err = %#v; wanted a *net.DNSError", err)
	} else if !de.IsTemporary {
		t.Fatalf("IsTemporary = false for err = %#v; want IsTemporary == true", err)
	}
}

func TestIssueNoSuchHostExists(t *testing.T) {
	err := lookupWithFake(fakeDNSServer{
		rh: func(n, _ string, q dnsmessage.Message, _ time.Time) (dnsmessage.Message, error) {
			return dnsmessage.Message{
				Header: dnsmessage.Header{
					ID:       q.ID,
					Response: true,
					RCode:    dnsmessage.RCodeNameError,
				},
				Questions: q.Questions,
			}, nil
		},
	}, "golang.org.", dnsmessage.TypeALL)
	if err == nil {
		t.Fatal("expected an error")
	}
	if _, ok := err.(Error); !ok {
		t.Fatalf("err = %#v; wanted something supporting net.Error", err)
	}
	if de, ok := err.(*DNSError); !ok {
		t.Fatalf("err = %#v; wanted a *net.DNSError", err)
	} else if !de.IsNotFound {
		t.Fatalf("IsNotFound = false for err = %#v; want IsNotFound == true", err)
	}
}

// TestNoSuchHost verifies that tryOneName works correctly when the domain does
// not exist.
//
// Issue 12778: verify that NXDOMAIN without RA bit errors as "no such host"
// and not "server misbehaving"
//
// Issue 25336: verify that NXDOMAIN errors fail fast.
//
// Issue 27525: verify that empty answers fail fast.
func TestNoSuchHost(t *testing.T) {
	tests := []struct {
		name string
		f    func(string, string, dnsmessage.Message, time.Time) (dnsmessage.Message, error)
	}{
		{
			"NXDOMAIN",
			func(n, _ string, q dnsmessage.Message, _ time.Time) (dnsmessage.Message, error) {
				return dnsmessage.Message{
					Header: dnsmessage.Header{
						ID:                 q.ID,
						Response:           true,
						RCode:              dnsmessage.RCodeNameError,
						RecursionAvailable: false,
					},
					Questions: q.Questions,
				}, nil
			},
		},
		{
			"no answers",
			func(n, _ string, q dnsmessage.Message, _ time.Time) (dnsmessage.Message, error) {
				return dnsmessage.Message{
					Header: dnsmessage.Header{
						ID:                 q.ID,
						Response:           true,
						RCode:              dnsmessage.RCodeSuccess,
						RecursionAvailable: false,
						Authoritative:      true,
					},
					Questions: q.Questions,
				}, nil
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			lookups := 0
			err := lookupWithFake(fakeDNSServer{
				rh: func(n, s string, q dnsmessage.Message, d time.Time) (dnsmessage.Message, error) {
					lookups++
					return test.f(n, s, q, d)
				},
			}, ".", dnsmessage.TypeALL)

			if lookups != 1 {
				t.Errorf("got %d lookups, wanted 1", lookups)
			}

			if err == nil {
				t.Fatal("expected an error")
			}
			de, ok := err.(*DNSError)
			if !ok {
				t.Fatalf("err = %#v; wanted a *net.DNSError", err)
			}
			if de.Err != errNoSuchHost.Error() {
				t.Fatalf("Err = %#v; wanted %q", de.Err, errNoSuchHost.Error())
			}
			if !de.IsNotFound {
				t.Fatalf("IsNotFound = %v wanted true", de.IsNotFound)
			}
		})
	}
}

// Issue 26573: verify that Conns that don't implement PacketConn are treated
// as streams even when udp was requested.
func TestDNSDialTCP(t *testing.T) {
	fake := fakeDNSServer{
		rh: func(n, _ string, q dnsmessage.Message, _ time.Time) (dnsmessage.Message, error) {
			r := dnsmessage.Message{
				Header: dnsmessage.Header{
					ID:       q.Header.ID,
					Response: true,
					RCode:    dnsmessage.RCodeSuccess,
				},
				Questions: q.Questions,
			}
			return r, nil
		},
		alwaysTCP: true,
	}
	r := Resolver{PreferGo: true, Dial: fake.DialContext}
	ctx := context.Background()
	_, _, err := r.exchange(ctx, "0.0.0.0", mustQuestion("com.", dnsmessage.TypeALL, dnsmessage.ClassINET), time.Second, useUDPOrTCP)
	if err != nil {
		t.Fatal("exhange failed:", err)
	}
}

// Issue 27763: verify that two strings in one TXT record are concatenated.
func TestTXTRecordTwoStrings(t *testing.T) {
	fake := fakeDNSServer{
		rh: func(n, _ string, q dnsmessage.Message, _ time.Time) (dnsmessage.Message, error) {
			r := dnsmessage.Message{
				Header: dnsmessage.Header{
					ID:       q.Header.ID,
					Response: true,
					RCode:    dnsmessage.RCodeSuccess,
				},
				Questions: q.Questions,
				Answers: []dnsmessage.Resource{
					{
						Header: dnsmessage.ResourceHeader{
							Name:  q.Questions[0].Name,
							Type:  dnsmessage.TypeA,
							Class: dnsmessage.ClassINET,
						},
						Body: &dnsmessage.TXTResource{
							TXT: []string{"string1 ", "string2"},
						},
					},
					{
						Header: dnsmessage.ResourceHeader{
							Name:  q.Questions[0].Name,
							Type:  dnsmessage.TypeA,
							Class: dnsmessage.ClassINET,
						},
						Body: &dnsmessage.TXTResource{
							TXT: []string{"onestring"},
						},
					},
				},
			}
			return r, nil
		},
	}
	r := Resolver{PreferGo: true, Dial: fake.DialContext}
	txt, err := r.lookupTXT(context.Background(), "golang.org")
	if err != nil {
		t.Fatal("LookupTXT failed:", err)
	}
	if want := 2; len(txt) != want {
		t.Fatalf("len(txt), got %d, want %d", len(txt), want)
	}
	if want := "string1 string2"; txt[0] != want {
		t.Errorf("txt[0], got %q, want %q", txt[0], want)
	}
	if want := "onestring"; txt[1] != want {
		t.Errorf("txt[1], got %q, want %q", txt[1], want)
	}
}

// Issue 29644: support single-request resolv.conf option in pure Go resolver.
// The A and AAAA queries will be sent sequentially, not in parallel.
func TestSingleRequestLookup(t *testing.T) {
	defer dnsWaitGroup.Wait()
	var (
		firstcalled int32
		ipv4        int32 = 1
		ipv6        int32 = 2
	)
	fake := fakeDNSServer{rh: func(n, s string, q dnsmessage.Message, _ time.Time) (dnsmessage.Message, error) {
		r := dnsmessage.Message{
			Header: dnsmessage.Header{
				ID:       q.ID,
				Response: true,
			},
			Questions: q.Questions,
		}
		for _, question := range q.Questions {
			switch question.Type {
			case dnsmessage.TypeA:
				if question.Name.String() == "slowipv4.example.net." {
					time.Sleep(10 * time.Millisecond)
				}
				if !atomic.CompareAndSwapInt32(&firstcalled, 0, ipv4) {
					t.Errorf("the A query was received after the AAAA query !")
				}
				r.Answers = append(r.Answers, dnsmessage.Resource{
					Header: dnsmessage.ResourceHeader{
						Name:   q.Questions[0].Name,
						Type:   dnsmessage.TypeA,
						Class:  dnsmessage.ClassINET,
						Length: 4,
					},
					Body: &dnsmessage.AResource{
						A: TestAddr,
					},
				})
			case dnsmessage.TypeAAAA:
				atomic.CompareAndSwapInt32(&firstcalled, 0, ipv6)
				r.Answers = append(r.Answers, dnsmessage.Resource{
					Header: dnsmessage.ResourceHeader{
						Name:   q.Questions[0].Name,
						Type:   dnsmessage.TypeAAAA,
						Class:  dnsmessage.ClassINET,
						Length: 16,
					},
					Body: &dnsmessage.AAAAResource{
						AAAA: TestAddr6,
					},
				})
			}
		}
		return r, nil
	}}
	r := Resolver{PreferGo: true, Dial: fake.DialContext}

	conf, err := newResolvConfTest()
	if err != nil {
		t.Fatal(err)
	}
	defer conf.teardown()
	if err := conf.writeAndUpdate([]string{"options single-request"}); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"hostname.example.net", "slowipv4.example.net"} {
		firstcalled = 0
		_, err := r.LookupIPAddr(context.Background(), name)
		if err != nil {
			t.Error(err)
		}
	}
}

// Issue 29358. Add configuration knob to force TCP-only DNS requests in the pure Go resolver.
func TestDNSUseTCP(t *testing.T) {
	fake := fakeDNSServer{
		rh: func(n, _ string, q dnsmessage.Message, _ time.Time) (dnsmessage.Message, error) {
			r := dnsmessage.Message{
				Header: dnsmessage.Header{
					ID:       q.Header.ID,
					Response: true,
					RCode:    dnsmessage.RCodeSuccess,
				},
				Questions: q.Questions,
			}
			if n == "udp" {
				t.Fatal("udp protocol was used instead of tcp")
			}
			return r, nil
		},
	}
	r := Resolver{PreferGo: true, Dial: fake.DialContext}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	_, _, err := r.exchange(ctx, "0.0.0.0", mustQuestion("com.", dnsmessage.TypeALL, dnsmessage.ClassINET), time.Second, useTCPOnly)
	if err != nil {
		t.Fatal("exchange failed:", err)
	}
}

// Issue 34660: PTR response with non-PTR answers should ignore non-PTR
func TestPTRandNonPTR(t *testing.T) {
	fake := fakeDNSServer{
		rh: func(n, _ string, q dnsmessage.Message, _ time.Time) (dnsmessage.Message, error) {
			r := dnsmessage.Message{
				Header: dnsmessage.Header{
					ID:       q.Header.ID,
					Response: true,
					RCode:    dnsmessage.RCodeSuccess,
				},
				Questions: q.Questions,
				Answers: []dnsmessage.Resource{
					{
						Header: dnsmessage.ResourceHeader{
							Name:  q.Questions[0].Name,
							Type:  dnsmessage.TypePTR,
							Class: dnsmessage.ClassINET,
						},
						Body: &dnsmessage.PTRResource{
							PTR: dnsmessage.MustNewName("golang.org."),
						},
					},
					{
						Header: dnsmessage.ResourceHeader{
							Name:  q.Questions[0].Name,
							Type:  dnsmessage.TypeTXT,
							Class: dnsmessage.ClassINET,
						},
						Body: &dnsmessage.TXTResource{
							TXT: []string{"PTR 8 6 60 ..."}, // fake RRSIG
						},
					},
				},
			}
			return r, nil
		},
	}
	r := Resolver{PreferGo: true, Dial: fake.DialContext}
	names, err := r.lookupAddr(context.Background(), "192.0.2.123")
	if err != nil {
		t.Fatalf("LookupAddr: %v", err)
	}
	if want := []string{"golang.org."}; !reflect.DeepEqual(names, want) {
		t.Errorf("names = %q; want %q", names, want)
	}
}

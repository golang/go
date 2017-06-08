// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package net

import (
	"context"
	"errors"
	"fmt"
	"internal/poll"
	"io/ioutil"
	"os"
	"path"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

var goResolver = Resolver{PreferGo: true}

// Test address from 192.0.2.0/24 block, reserved by RFC 5737 for documentation.
const TestAddr uint32 = 0xc0000201

// Test address from 2001:db8::/32 block, reserved by RFC 3849 for documentation.
var TestAddr6 = [16]byte{0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}

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
	fake := fakeDNSServer{
		rh: func(n, _ string, q *dnsMsg, _ time.Time) (*dnsMsg, error) {
			r := &dnsMsg{
				dnsMsgHdr: dnsMsgHdr{
					id:       q.id,
					response: true,
					rcode:    dnsRcodeSuccess,
				},
				question: q.question,
			}
			if n == "udp" {
				r.truncated = true
			}
			return r, nil
		},
	}
	r := Resolver{PreferGo: true, Dial: fake.DialContext}
	for _, tt := range dnsTransportFallbackTests {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		msg, err := r.exchange(ctx, tt.server, tt.name, tt.qtype, time.Second)
		if err != nil {
			t.Error(err)
			continue
		}
		switch msg.rcode {
		case tt.rcode:
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
	fake := fakeDNSServer{func(_, _ string, q *dnsMsg, _ time.Time) (*dnsMsg, error) {
		r := &dnsMsg{
			dnsMsgHdr: dnsMsgHdr{
				id:       q.id,
				response: true,
			},
			question: q.question,
		}

		switch q.question[0].Name {
		case "example.com.":
			r.rcode = dnsRcodeSuccess
		default:
			r.rcode = dnsRcodeNameError
		}

		return r, nil
	}}
	r := Resolver{PreferGo: true, Dial: fake.DialContext}
	server := "8.8.8.8:53"
	for _, tt := range specialDomainNameTests {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		msg, err := r.exchange(ctx, server, tt.name, tt.qtype, 3*time.Second)
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

var fakeDNSServerSuccessful = fakeDNSServer{func(_, _ string, q *dnsMsg, _ time.Time) (*dnsMsg, error) {
	r := &dnsMsg{
		dnsMsgHdr: dnsMsgHdr{
			id:       q.id,
			response: true,
		},
		question: q.question,
	}
	if len(q.question) == 1 && q.question[0].Qtype == dnsTypeA {
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
	return r, nil
}}

// Issue 13705: don't try to resolve onion addresses, etc
func TestLookupTorOnion(t *testing.T) {
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
	fake := fakeDNSServer{func(n, s string, q *dnsMsg, _ time.Time) (*dnsMsg, error) {
		switch s {
		case "[2001:4860:4860::8888]:53", "8.8.8.8:53":
			break
		default:
			time.Sleep(10 * time.Millisecond)
			return nil, poll.ErrTimeout
		}
		r := &dnsMsg{
			dnsMsgHdr: dnsMsgHdr{
				id:       q.id,
				response: true,
			},
			question: q.question,
		}
		for _, question := range q.question {
			switch question.Qtype {
			case dnsTypeA:
				switch question.Name {
				case "hostname.as112.net.":
					break
				case "ipv4.google.com.":
					r.answer = append(r.answer, &dnsRR_A{
						Hdr: dnsRR_Header{
							Name:     q.question[0].Name,
							Rrtype:   dnsTypeA,
							Class:    dnsClassINET,
							Rdlength: 4,
						},
						A: TestAddr,
					})
				default:

				}
			case dnsTypeAAAA:
				switch question.Name {
				case "hostname.as112.net.":
					break
				case "ipv6.google.com.":
					r.answer = append(r.answer, &dnsRR_AAAA{
						Hdr: dnsRR_Header{
							Name:     q.question[0].Name,
							Rrtype:   dnsTypeAAAA,
							Class:    dnsClassINET,
							Rdlength: 16,
						},
						AAAA: TestAddr6,
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
	fake := fakeDNSServer{func(n, s string, q *dnsMsg, tm time.Time) (*dnsMsg, error) {
		r := &dnsMsg{
			dnsMsgHdr: dnsMsgHdr{
				id:       q.id,
				response: true,
			},
			question: q.question,
		}
		return r, nil
	}}
	r := Resolver{PreferGo: true, Dial: fake.DialContext}

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
	defer conf.teardown()
}

// Issue 12712.
// When using search domains, return the error encountered
// querying the original name instead of an error encountered
// querying a generated name.
func TestErrorForOriginalNameWhenSearching(t *testing.T) {
	const fqdn = "doesnotexist.domain"

	conf, err := newResolvConfTest()
	if err != nil {
		t.Fatal(err)
	}
	defer conf.teardown()

	if err := conf.writeAndUpdate([]string{"search servfail"}); err != nil {
		t.Fatal(err)
	}

	fake := fakeDNSServer{func(_, _ string, q *dnsMsg, _ time.Time) (*dnsMsg, error) {
		r := &dnsMsg{
			dnsMsgHdr: dnsMsgHdr{
				id:       q.id,
				response: true,
			},
			question: q.question,
		}

		switch q.question[0].Name {
		case fqdn + ".servfail.":
			r.rcode = dnsRcodeServerFailure
		default:
			r.rcode = dnsRcodeNameError
		}

		return r, nil
	}}

	cases := []struct {
		strictErrors bool
		wantErr      *DNSError
	}{
		{true, &DNSError{Name: fqdn, Err: "server misbehaving", IsTemporary: true}},
		{false, &DNSError{Name: fqdn, Err: errNoSuchHost.Error()}},
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
	conf, err := newResolvConfTest()
	if err != nil {
		t.Fatal(err)
	}
	defer conf.teardown()

	if err := conf.writeAndUpdate([]string{"nameserver 192.0.2.1", // the one that will give a lame referral
		"nameserver 192.0.2.2"}); err != nil {
		t.Fatal(err)
	}

	fake := fakeDNSServer{func(_, s string, q *dnsMsg, _ time.Time) (*dnsMsg, error) {
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

	for i := 0; i < b.N; i++ {
		goResolver.LookupIPAddr(ctx, "www.example.com")
	}
}

func BenchmarkGoLookupIPNoSuchHost(b *testing.B) {
	testHookUninstaller.Do(uninstallTestHooks)
	ctx := context.Background()

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

	for i := 0; i < b.N; i++ {
		goResolver.LookupIPAddr(ctx, "www.example.com")
	}
}

type fakeDNSServer struct {
	rh func(n, s string, q *dnsMsg, t time.Time) (*dnsMsg, error)
}

func (server *fakeDNSServer) DialContext(_ context.Context, n, s string) (Conn, error) {
	return &fakeDNSConn{nil, server, n, s, nil, time.Time{}}, nil
}

type fakeDNSConn struct {
	Conn
	server *fakeDNSServer
	n      string
	s      string
	q      *dnsMsg
	t      time.Time
}

func (f *fakeDNSConn) Close() error {
	return nil
}

func (f *fakeDNSConn) Read(b []byte) (int, error) {
	resp, err := f.server.rh(f.n, f.s, f.q, f.t)
	if err != nil {
		return 0, err
	}

	bb, ok := resp.Pack()
	if !ok {
		return 0, errors.New("cannot marshal DNS message")
	}
	if len(b) < len(bb) {
		return 0, errors.New("read would fragment DNS message")
	}

	copy(b, bb)
	return len(bb), nil
}

func (f *fakeDNSConn) ReadFrom(b []byte) (int, Addr, error) {
	return 0, nil, nil
}

func (f *fakeDNSConn) Write(b []byte) (int, error) {
	f.q = new(dnsMsg)
	if !f.q.Unpack(b) {
		return 0, errors.New("cannot unmarshal DNS message")
	}
	return len(b), nil
}

func (f *fakeDNSConn) WriteTo(b []byte, addr Addr) (int, error) {
	return 0, nil
}

func (f *fakeDNSConn) SetDeadline(t time.Time) error {
	f.t = t
	return nil
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

		msg := &dnsMsg{}
		if !msg.Unpack(b[:n]) {
			t.Error("invalid DNS query")
			return
		}

		s.Write([]byte("garbage DNS response packet"))

		msg.response = true
		msg.id++ // make invalid ID
		b, ok := msg.Pack()
		if !ok {
			t.Error("failed to pack DNS response")
			return
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
			t.Error("failed to pack DNS response")
			return
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

	dc := &dnsPacketConn{c}
	resp, err := dc.dnsRoundTrip(msg)
	if err != nil {
		t.Fatalf("dnsRoundTripUDP failed: %v", err)
	}

	if got := resp.answer[0].(*dnsRR_A).A; got != TestAddr {
		t.Errorf("got address %v, want %v", got, TestAddr)
	}
}

// Issue 16865. If a name server times out, continue to the next.
func TestRetryTimeout(t *testing.T) {
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

	fake := fakeDNSServer{func(_, s string, q *dnsMsg, deadline time.Time) (*dnsMsg, error) {
		t.Log(s, q, deadline)

		if deadline.IsZero() {
			t.Error("zero deadline")
		}

		if s == "192.0.2.1:53" {
			deadline0 = deadline
			time.Sleep(10 * time.Millisecond)
			return nil, poll.ErrTimeout
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
	fake := fakeDNSServer{func(_, s string, q *dnsMsg, deadline time.Time) (*dnsMsg, error) {
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

func mockTXTResponse(q *dnsMsg) *dnsMsg {
	r := &dnsMsg{
		dnsMsgHdr: dnsMsgHdr{
			id:                  q.id,
			response:            true,
			recursion_available: true,
		},
		question: q.question,
		answer: []dnsRR{
			&dnsRR_TXT{
				Hdr: dnsRR_Header{
					Name:   q.question[0].Name,
					Rrtype: dnsTypeTXT,
					Class:  dnsClassINET,
				},
				Txt: "ok",
			},
		},
	}

	return r
}

// Issue 17448. With StrictErrors enabled, temporary errors should make
// LookupIP fail rather than return a partial result.
func TestStrictErrorsLookupIP(t *testing.T) {
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
			Err:       poll.ErrTimeout.Error(),
			Name:      name,
			Server:    server,
			IsTimeout: true,
		}
	}
	makeNxDomain := func() error {
		return &DNSError{
			Err:    errNoSuchHost.Error(),
			Name:   name,
			Server: server,
		}
	}

	cases := []struct {
		desc          string
		resolveWhich  func(quest *dnsQuestion) resolveWhichEnum
		wantStrictErr error
		wantLaxErr    error
		wantIPs       []string
	}{
		{
			desc: "No errors",
			resolveWhich: func(quest *dnsQuestion) resolveWhichEnum {
				return resolveOK
			},
			wantIPs: []string{ip4, ip6},
		},
		{
			desc: "searchX error fails in strict mode",
			resolveWhich: func(quest *dnsQuestion) resolveWhichEnum {
				if quest.Name == searchX {
					return resolveTimeout
				}
				return resolveOK
			},
			wantStrictErr: makeTimeout(),
			wantIPs:       []string{ip4, ip6},
		},
		{
			desc: "searchX IPv4-only timeout fails in strict mode",
			resolveWhich: func(quest *dnsQuestion) resolveWhichEnum {
				if quest.Name == searchX && quest.Qtype == dnsTypeA {
					return resolveTimeout
				}
				return resolveOK
			},
			wantStrictErr: makeTimeout(),
			wantIPs:       []string{ip4, ip6},
		},
		{
			desc: "searchX IPv6-only servfail fails in strict mode",
			resolveWhich: func(quest *dnsQuestion) resolveWhichEnum {
				if quest.Name == searchX && quest.Qtype == dnsTypeAAAA {
					return resolveServfail
				}
				return resolveOK
			},
			wantStrictErr: makeTempError("server misbehaving"),
			wantIPs:       []string{ip4, ip6},
		},
		{
			desc: "searchY error always fails",
			resolveWhich: func(quest *dnsQuestion) resolveWhichEnum {
				if quest.Name == searchY {
					return resolveTimeout
				}
				return resolveOK
			},
			wantStrictErr: makeTimeout(),
			wantLaxErr:    makeNxDomain(), // This one reaches the "test." FQDN.
		},
		{
			desc: "searchY IPv4-only socket error fails in strict mode",
			resolveWhich: func(quest *dnsQuestion) resolveWhichEnum {
				if quest.Name == searchY && quest.Qtype == dnsTypeA {
					return resolveOpError
				}
				return resolveOK
			},
			wantStrictErr: makeTempError("write: socket on fire"),
			wantIPs:       []string{ip6},
		},
		{
			desc: "searchY IPv6-only timeout fails in strict mode",
			resolveWhich: func(quest *dnsQuestion) resolveWhichEnum {
				if quest.Name == searchY && quest.Qtype == dnsTypeAAAA {
					return resolveTimeout
				}
				return resolveOK
			},
			wantStrictErr: makeTimeout(),
			wantIPs:       []string{ip4},
		},
	}

	for i, tt := range cases {
		fake := fakeDNSServer{func(_, s string, q *dnsMsg, deadline time.Time) (*dnsMsg, error) {
			t.Log(s, q)

			switch tt.resolveWhich(&q.question[0]) {
			case resolveOK:
				// Handle below.
			case resolveOpError:
				return nil, &OpError{Op: "write", Err: fmt.Errorf("socket on fire")}
			case resolveServfail:
				return &dnsMsg{
					dnsMsgHdr: dnsMsgHdr{
						id:       q.id,
						response: true,
						rcode:    dnsRcodeServerFailure,
					},
					question: q.question,
				}, nil
			case resolveTimeout:
				return nil, poll.ErrTimeout
			default:
				t.Fatal("Impossible resolveWhich")
			}

			switch q.question[0].Name {
			case searchX, name + ".":
				// Return NXDOMAIN to utilize the search list.
				return &dnsMsg{
					dnsMsgHdr: dnsMsgHdr{
						id:       q.id,
						response: true,
						rcode:    dnsRcodeNameError,
					},
					question: q.question,
				}, nil
			case searchY:
				// Return records below.
			default:
				return nil, fmt.Errorf("Unexpected Name: %v", q.question[0].Name)
			}

			r := &dnsMsg{
				dnsMsgHdr: dnsMsgHdr{
					id:       q.id,
					response: true,
				},
				question: q.question,
			}
			switch q.question[0].Qtype {
			case dnsTypeA:
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
			case dnsTypeAAAA:
				r.answer = []dnsRR{
					&dnsRR_AAAA{
						Hdr: dnsRR_Header{
							Name:     q.question[0].Name,
							Rrtype:   dnsTypeAAAA,
							Class:    dnsClassINET,
							Rdlength: 16,
						},
						AAAA: TestAddr6,
					},
				}
			default:
				return nil, fmt.Errorf("Unexpected Qtype: %v", q.question[0].Qtype)
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

	fake := fakeDNSServer{func(_, s string, q *dnsMsg, deadline time.Time) (*dnsMsg, error) {
		t.Log(s, q)

		switch q.question[0].Name {
		case searchX:
			return nil, poll.ErrTimeout
		case searchY:
			return mockTXTResponse(q), nil
		default:
			return nil, fmt.Errorf("Unexpected Name: %v", q.question[0].Name)
		}
	}}

	for _, strict := range []bool{true, false} {
		r := Resolver{StrictErrors: strict, Dial: fake.DialContext}
		_, rrs, err := r.lookup(context.Background(), name, dnsTypeTXT)
		var wantErr error
		var wantRRs int
		if strict {
			wantErr = &DNSError{
				Err:       poll.ErrTimeout.Error(),
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
		if len(rrs) != wantRRs {
			t.Errorf("strict=%v: got %v; want %v", strict, len(rrs), wantRRs)
		}
	}
}

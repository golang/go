// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package net

import (
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

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
	if testing.Short() || !*testExternal {
		t.Skip("avoid external network")
	}

	for _, tt := range dnsTransportFallbackTests {
		timeout := time.Duration(tt.timeout) * time.Second
		msg, err := exchange(tt.server, tt.name, tt.qtype, timeout)
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
	// Though, we test those names here for verifying nagative
	// answers at DNS query-response interaction level.
	{"localhost.", dnsTypeALL, dnsRcodeNameError},
	{"invalid.", dnsTypeALL, dnsRcodeNameError},
}

func TestSpecialDomainName(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("avoid external network")
	}

	server := "8.8.8.8:53"
	for _, tt := range specialDomainNameTests {
		msg, err := exchange(server, tt.name, tt.qtype, 0)
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
	if err := conf.forceUpdate(conf.path); err != nil {
		return err
	}
	return nil
}

func (conf *resolvConfTest) forceUpdate(name string) error {
	dnsConf := dnsReadConfig(name)
	conf.mu.Lock()
	conf.dnsConfig = dnsConf
	conf.mu.Unlock()
	for i := 0; i < 5; i++ {
		if conf.tryAcquireSema() {
			conf.lastChecked = time.Time{}
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
	err := conf.forceUpdate("/etc/resolv.conf")
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
		servers: []string{"8.8.8.8"},
	},
	{
		name:    "",
		lines:   nil, // an empty resolv.conf should use defaultNS as name servers
		servers: defaultNS,
	},
	{
		name:    "www.example.com",
		lines:   []string{"nameserver 8.8.4.4"},
		servers: []string{"8.8.4.4"},
	},
}

func TestUpdateResolvConf(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("avoid external network")
	}

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
					ips, err := goLookupIP(name)
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
	if testing.Short() || !*testExternal {
		t.Skip("avoid external network")
	}

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
		conf.tryUpdate(conf.path)
		addrs, err := goLookupIP(tt.name)
		if err != nil {
			if err, ok := err.(*DNSError); !ok || (err.Name != tt.error.(*DNSError).Name || err.Server != tt.error.(*DNSError).Server || err.IsTimeout != tt.error.(*DNSError).IsTimeout) {
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
	if testing.Short() || !*testExternal {
		t.Skip("avoid external network")
	}

	// Add a config that simulates no dns servers being available.
	conf, err := newResolvConfTest()
	if err != nil {
		t.Fatal(err)
	}
	if err := conf.writeAndUpdate([]string{}); err != nil {
		t.Fatal(err)
	}
	conf.tryUpdate(conf.path)
	// Redirect host file lookups.
	defer func(orig string) { testHookHostsPath = orig }(testHookHostsPath)
	testHookHostsPath = "testdata/hosts"

	for _, order := range []hostLookupOrder{hostLookupFilesDNS, hostLookupDNSFiles} {
		name := fmt.Sprintf("order %v", order)

		// First ensure that we get an error when contacting a non-existant host.
		_, err := goLookupIPOrder("notarealhost", order)
		if err == nil {
			t.Errorf("%s: expected error while looking up name not in hosts file", name)
			continue
		}

		// Now check that we get an address when the name appears in the hosts file.
		addrs, err := goLookupIPOrder("thor", order) // entry is in "testdata/hosts"
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

	d := &fakeDNSConn{}
	testHookDNSDialer = func(time.Duration) dnsDialer { return d }

	d.rh = func(q *dnsMsg) (*dnsMsg, error) {
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

	_, err = goLookupIP(fqdn)
	if err == nil {
		t.Fatal("expected an error")
	}

	want := &DNSError{Name: fqdn, Err: errNoSuchHost.Error()}
	if err, ok := err.(*DNSError); !ok || err.Name != want.Name || err.Err != want.Err {
		t.Errorf("got %v; want %v", err, want)
	}
}

func BenchmarkGoLookupIP(b *testing.B) {
	testHookUninstaller.Do(uninstallTestHooks)

	for i := 0; i < b.N; i++ {
		goLookupIP("www.example.com")
	}
}

func BenchmarkGoLookupIPNoSuchHost(b *testing.B) {
	testHookUninstaller.Do(uninstallTestHooks)

	for i := 0; i < b.N; i++ {
		goLookupIP("some.nonexistent")
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

	for i := 0; i < b.N; i++ {
		goLookupIP("www.example.com")
	}
}

type fakeDNSConn struct {
	// last query
	qmu sync.Mutex // guards q
	q   *dnsMsg
	// reply handler
	rh func(*dnsMsg) (*dnsMsg, error)
}

func (f *fakeDNSConn) dialDNS(n, s string) (dnsConn, error) {
	return f, nil
}

func (f *fakeDNSConn) Close() error {
	return nil
}

func (f *fakeDNSConn) SetDeadline(time.Time) error {
	return nil
}

func (f *fakeDNSConn) writeDNSQuery(q *dnsMsg) error {
	f.qmu.Lock()
	defer f.qmu.Unlock()
	f.q = q
	return nil
}

func (f *fakeDNSConn) readDNSResponse() (*dnsMsg, error) {
	f.qmu.Lock()
	q := f.q
	f.qmu.Unlock()
	return f.rh(q)
}
